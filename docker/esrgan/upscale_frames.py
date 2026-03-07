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
    return parser.parse_args()


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
    image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read frame: {frame_path}")
    image = image.astype(np.float32) / 255.0
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_half = device.type == "cuda"
    if device.type == "cuda":
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
            output = (output * 255.0).round().astype(np.uint8)

            out_path = os.path.join(args.output_dir, os.path.basename(frame_path))
            ok = cv2.imwrite(out_path, output)
            if not ok:
                raise RuntimeError(f"Could not write upscaled frame: {out_path}")

        pbar.update(len(batch_paths))
    pbar.close()


if __name__ == "__main__":
    main()
