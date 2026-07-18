from argparse import ArgumentParser
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
import numpy as np
import torch  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

import depth_pro  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]


DEPTH_PRO_CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"


def _ensure_checkpoint(checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return
    print(f"Depth Pro checkpoint missing at {checkpoint_path}. Downloading...")
    urlretrieve(DEPTH_PRO_CHECKPOINT_URL, str(checkpoint_path))


def _resolve_precision(requested: str, device: torch.device) -> torch.dtype:
    if requested == "fp16":
        return torch.float16
    return torch.float32


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Depth Pro inference.")

    try:
        # Probe one tiny kernel to catch "no kernel image is available" at startup.
        _ = (torch.zeros(1, device="cuda") + 1).item()
        return torch.device("cuda:0")
    except Exception as exc:
        raise RuntimeError(
            "CUDA is visible but not usable by this PyTorch build. "
            f"Reason: {exc}"
        ) from exc


def _load_model(dtype: torch.dtype, device: torch.device):
    model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=dtype,
    )
    model.eval()
    return model, transform


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max > depth_min:
        normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        normalized = np.zeros_like(depth)
    return (normalized * 255.0).astype(np.uint8)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    input_video = Path(args.input_video).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise RuntimeError(f"Input video not found: {input_video}")

    # Depth Pro defaults to a relative checkpoint path (./checkpoints/depth_pro.pt).
    # Ensure cwd and checkpoint file are aligned even when /opt/ml-depth-pro/checkpoints is a host mount.
    os.chdir("/opt/ml-depth-pro")
    _ensure_checkpoint(Path("/opt/ml-depth-pro/checkpoints/depth_pro.pt"))

    device = _resolve_device()
    dtype = _resolve_precision(args.precision, device)

    model, transform = _load_model(dtype=dtype, device=device)

    depth_maps_dir = output_dir / "depth_maps"
    depth_maps_dir.mkdir(parents=True, exist_ok=True)
    depth_npz_dir = output_dir / "depth_npz"
    depth_npz_dir.mkdir(parents=True, exist_ok=True)

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
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb)
        prediction = model.infer(input_tensor, f_px=None)
        depth_tensor = prediction["depth"]
        depth = depth_tensor.detach().cpu().numpy().astype(np.float32)

        frame_count += 1
        npz_path = depth_npz_dir / f"depth_{frame_count:06d}.npz"
        np.savez_compressed(npz_path, depth=depth)

        depth_u8 = _normalize_depth(depth)
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
