from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Dict, List
from inspect import signature

import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from huggingface_hub import hf_hub_download


def _model_configs() -> Dict[str, Dict[str, object]]:
    return {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }


def _checkpoint_repo_candidates(encoder: str) -> List[str]:
    repo_by_encoder = {
        "vits": ["depth-anything/Depth-Anything-V2-Small"],
        "vitb": ["depth-anything/Depth-Anything-V2-Base"],
        "vitl": ["depth-anything/Depth-Anything-V2-Large"],
    }
    return repo_by_encoder[encoder]


def _resolve_checkpoint(encoder: str, checkpoint_root: Path) -> tuple[Path, str]:
    checkpoint_name = f"depth_anything_v2_{encoder}.pth"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if "DEPTH_ANYTHING_V2_CHECKPOINT_PATH" in os.environ:
        candidate = Path(os.environ["DEPTH_ANYTHING_V2_CHECKPOINT_PATH"]).expanduser()
        if candidate.exists():
            return candidate, encoder
        raise RuntimeError(f"Checkpoint path does not exist: {candidate}")

    local_checkpoint = checkpoint_root / checkpoint_name
    if local_checkpoint.exists():
        return local_checkpoint, encoder

    errors: List[str] = []
    for repo_id in _checkpoint_repo_candidates(encoder):
        try:
            effective_checkpoint_name = f"depth_anything_v2_{encoder}.pth"
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=effective_checkpoint_name,
                local_dir=str(checkpoint_root),
            )
            return Path(resolved), encoder
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")

    raise RuntimeError(
        "Failed to download Depth Anything V2 checkpoint for "
        f"encoder={encoder}. Tried repos: {_checkpoint_repo_candidates(encoder)}. "
        f"Errors: {' | '.join(errors)}"
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vitl")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--fast-resize-height", type=int, default=0)
    parser.add_argument("--temporal-smoothing-alpha", type=float, default=0.0)
    args = parser.parse_args()

    input_video = Path(args.input_video).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise RuntimeError(f"Input video not found: {input_video}")

    if args.input_size <= 0:
        raise RuntimeError("input_size must be positive")

    if args.fast_resize_height < 0:
        raise RuntimeError("fast_resize_height must be >= 0")

    if args.temporal_smoothing_alpha < 0.0 or args.temporal_smoothing_alpha > 1.0:
        raise RuntimeError("temporal_smoothing_alpha must be in [0.0, 1.0]")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Depth Anything V2 inference.")

    device = "cuda"
    checkpoint_path, effective_encoder = _resolve_checkpoint(
        args.encoder, Path("/opt/depth-anything-v2/checkpoints")
    )
    configs = _model_configs()
    model = DepthAnythingV2(**configs[effective_encoder])
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    model = model.to(device).eval()

    if args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    depth_maps_dir = output_dir / "depth_maps"
    depth_maps_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    preview_path = output_dir / "depth_preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    frame_count = 0
    previous_normalized = None
    infer_sig = signature(model.infer_image)
    supports_precision_arg = "precision" in infer_sig.parameters
    supports_newheight_arg = "newHeight" in infer_sig.parameters
    supports_newwidth_arg = "newWidth" in infer_sig.parameters

    def _infer_frame(frame: np.ndarray) -> np.ndarray:
        infer_kwargs = {}
        if supports_precision_arg:
            infer_kwargs["precision"] = args.precision

        effective_size = args.input_size
        if (
            args.fast_resize_height > 0
            and supports_newheight_arg
            and supports_newwidth_arg
        ):
            frame_height_local, frame_width_local = frame.shape[:2]
            aspect_ratio = frame_width_local / frame_height_local
            target_height = args.fast_resize_height
            target_width = round((target_height * aspect_ratio) / 14) * 14
            target_width = max(14, (target_width // 14) * 14)
            infer_kwargs["newHeight"] = target_height
            infer_kwargs["newWidth"] = target_width
            effective_size = min(effective_size, max(target_height, target_width))

        if args.precision == "fp16" and device == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_context = torch.autocast(device_type="cpu", enabled=False)

        with autocast_context:
            depth_pred = model.infer_image(frame, effective_size, **infer_kwargs)

        if isinstance(depth_pred, torch.Tensor):
            depth_pred = depth_pred.detach().float().cpu().numpy()
        return depth_pred

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        try:
            depth = _infer_frame(frame)
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything V2 inference failed. "
                f"input_size={args.input_size}, precision={args.precision}. "
                f"Original error: {exc}"
            ) from exc

        depth_min = float(depth.min())
        depth_max = float(depth.max())

        if depth_max > depth_min:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth)

        if args.temporal_smoothing_alpha > 0.0 and previous_normalized is not None:
            alpha = args.temporal_smoothing_alpha
            normalized = (alpha * previous_normalized) + ((1.0 - alpha) * normalized)

        previous_normalized = normalized

        depth_u8 = (normalized * 255.0).astype(np.uint8)
        frame_count += 1

        png_path = depth_maps_dir / f"depth_{frame_count:06d}.png"
        cv2.imwrite(str(png_path), depth_u8)

        preview = np.repeat(depth_u8[:, :, np.newaxis], 3, axis=2)
        writer.write(preview)

    cap.release()
    writer.release()

    if frame_count == 0:
        raise RuntimeError("No frames were processed from input video.")


if __name__ == "__main__":
    main()
