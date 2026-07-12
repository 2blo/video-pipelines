from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3


MODEL_IDS: Dict[str, str] = {
    "small": "depth-anything/DA3-SMALL",
}


def _resize_long_edge(frame_rgb: np.ndarray, max_res: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_res:
        return frame_rgb

    scale = max_res / float(longest_edge)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(
        frame_rgb,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def _depth_to_u8(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

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
    parser.add_argument("--model", choices=sorted(MODEL_IDS.keys()), default="small")
    parser.add_argument("--max-res", type=int, default=1280)
    args = parser.parse_args()

    if args.max_res <= 0:
        raise RuntimeError("max_res must be positive")

    input_video = Path(args.input_video).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise RuntimeError(f"Input video not found: {input_video}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(MODEL_IDS[args.model])
    model = model.to(device=device)

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
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = _resize_long_edge(frame_rgb, args.max_res)

            prediction = model.inference([resized_rgb])
            depth_prediction = np.asarray(prediction.depth[0], dtype=np.float32)
            depth_u8 = _depth_to_u8(depth_prediction)

            if depth_u8.shape[1] != frame_width or depth_u8.shape[0] != frame_height:
                depth_u8 = cv2.resize(
                    depth_u8,
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_LINEAR,
                )

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
