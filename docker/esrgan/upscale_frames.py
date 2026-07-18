import argparse
import glob
import os
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from RRDBNet_arch import RRDBNet


def remap_realesrgan_keys(load_net: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in load_net.items():
        new_key = key

        if key.startswith("body."):
            new_key = key.replace("body.", "RRDB_trunk.", 1)
            new_key = new_key.replace(".rdb1.", ".RDB1.")
            new_key = new_key.replace(".rdb2.", ".RDB2.")
            new_key = new_key.replace(".rdb3.", ".RDB3.")

        if new_key.startswith("conv_body."):
            new_key = new_key.replace("conv_body.", "trunk_conv.", 1)
        elif new_key.startswith("conv_up1."):
            new_key = new_key.replace("conv_up1.", "upconv1.", 1)
        elif new_key.startswith("conv_up2."):
            new_key = new_key.replace("conv_up2.", "upconv2.", 1)
        elif new_key.startswith("conv_hr."):
            new_key = new_key.replace("conv_hr.", "HRconv.", 1)

        remapped[new_key] = value

    return remapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=0)
    parser.add_argument("--output-format", choices=["png", "exr"], default="png")
    parser.add_argument("--exr-type", choices=["half", "float"], default="float")
    parser.add_argument(
        "--exr-compression",
        choices=[
            "none",
            "rle",
            "zips",
            "zip",
            "piz",
            "pxr24",
            "b44",
            "b44a",
            "dwaa",
            "dwab",
        ],
        default="zip",
    )
    return parser.parse_args()


def _resolve_exr_imwrite_params(exr_type: str, compression: str) -> List[int]:
    compression_attr = {
        "none": "IMWRITE_EXR_COMPRESSION_NO",
        "rle": "IMWRITE_EXR_COMPRESSION_RLE",
        "zips": "IMWRITE_EXR_COMPRESSION_ZIPS",
        "zip": "IMWRITE_EXR_COMPRESSION_ZIP",
        "piz": "IMWRITE_EXR_COMPRESSION_PIZ",
        "pxr24": "IMWRITE_EXR_COMPRESSION_PXR24",
        "b44": "IMWRITE_EXR_COMPRESSION_B44",
        "b44a": "IMWRITE_EXR_COMPRESSION_B44A",
        "dwaa": "IMWRITE_EXR_COMPRESSION_DWAA",
        "dwab": "IMWRITE_EXR_COMPRESSION_DWAB",
    }
    type_attr = {
        "half": "IMWRITE_EXR_TYPE_HALF",
        "float": "IMWRITE_EXR_TYPE_FLOAT",
    }

    required_attrs = [
        "IMWRITE_EXR_COMPRESSION",
        compression_attr[compression],
        "IMWRITE_EXR_TYPE",
        type_attr[exr_type],
    ]
    missing = [attr for attr in required_attrs if not hasattr(cv2, attr)]
    if missing:
        raise RuntimeError(
            "OpenCV EXR writing support is missing required constants: "
            + ", ".join(missing)
        )

    return [
        int(getattr(cv2, "IMWRITE_EXR_COMPRESSION")),
        int(getattr(cv2, compression_attr[compression])),
        int(getattr(cv2, "IMWRITE_EXR_TYPE")),
        int(getattr(cv2, type_attr[exr_type])),
    ]


def load_model(model_path: str, device: torch.device, use_half: bool) -> RRDBNet:
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    try:
        load_obj: Dict[str, Any] = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        load_obj = torch.load(model_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load ESRGAN model file: {model_path}") from exc
    load_net: Dict[str, torch.Tensor]
    if isinstance(load_obj, dict) and "params_ema" in load_obj:
        load_net = load_obj["params_ema"]
    elif isinstance(load_obj, dict) and "state_dict" in load_obj:
        load_net = load_obj["state_dict"]
    elif isinstance(load_obj, dict):
        load_net = load_obj
    else:
        raise RuntimeError(f"Unexpected ESRGAN checkpoint format: {model_path}")

    try:
        model.load_state_dict(load_net, strict=True)
    except RuntimeError:
        model.load_state_dict(remap_realesrgan_keys(load_net), strict=True)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


def load_frame_tensor(frame_path: str) -> np.ndarray:
    image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read frame: {frame_path}")

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
    else:
        raise RuntimeError(
            f"Unsupported frame dtype for ESRGAN input: {image.dtype} at {frame_path}"
        )

    return np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))


def infer_batch_size(frame_path: str, requested_batch_size: int, use_half: bool) -> int:
    if requested_batch_size > 0:
        return requested_batch_size
    if not use_half:
        return 1
    sample = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if sample is None:
        return 1
    height, width, _ = sample.shape
    pixels = width * height
    if pixels <= (1280 * 720):
        return 4
    if pixels <= (1920 * 1080):
        return 2
    return 1


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.target_width < 0 or args.target_height < 0:
        raise RuntimeError("target-width and target-height must be non-negative.")
    apply_resize = args.target_width > 0 and args.target_height > 0

    effective_output_format = args.output_format
    if args.output_format == "exr" and not cv2.haveImageWriter(".exr"):
        effective_output_format = "png"
        print("OpenCV EXR writer unavailable; falling back to PNG output.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ESRGAN inference.")
    try:
        _ = (torch.zeros(1, device="cuda") + 1).item()
    except Exception as exc:
        raise RuntimeError(
            "CUDA is visible but unusable by this PyTorch build. "
            f"torch={torch.__version__}, cuda={torch.version.cuda}"
        ) from exc

    device = torch.device("cuda:0")
    use_half = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    frame_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {args.input_dir}")
    batch_size = infer_batch_size(frame_paths[0], args.batch_size, use_half)
    exr_params: List[int] | None = None
    if effective_output_format == "exr":
        exr_params = _resolve_exr_imwrite_params(args.exr_type, args.exr_compression)

    model = load_model(args.model_path, device, use_half=use_half)

    pbar = tqdm(total=len(frame_paths))
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        frames: List[np.ndarray] = [
            load_frame_tensor(frame_path) for frame_path in batch_paths
        ]
        batch = np.stack(frames, axis=0)
        input_tensor = torch.from_numpy(batch).to(
            device=device,
            dtype=torch.float16 if use_half else torch.float32,
            non_blocking=True,
        )

        with torch.inference_mode():
            output_batch = model(input_tensor).float().cpu().clamp_(0, 1).numpy()

        for index, frame_path in enumerate(batch_paths):
            output = output_batch[index]
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

            if apply_resize and (
                output.shape[1] != args.target_width
                or output.shape[0] != args.target_height
            ):
                output = cv2.resize(
                    output,
                    (args.target_width, args.target_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if effective_output_format == "png":
                output_u8 = (output * 255.0).round().astype(np.uint8)
                out_path = os.path.join(args.output_dir, os.path.basename(frame_path))
                ok = cv2.imwrite(out_path, output_u8)
            else:
                stem, _ = os.path.splitext(os.path.basename(frame_path))
                out_path = os.path.join(args.output_dir, f"{stem}.exr")
                if exr_params is None:
                    raise RuntimeError("EXR writer params are not initialized.")
                ok = cv2.imwrite(out_path, output.astype(np.float32), exr_params)
            if not ok:
                raise RuntimeError(f"Could not write upscaled frame: {out_path}")

        pbar.update(len(batch_paths))
    pbar.close()


if __name__ == "__main__":
    main()
