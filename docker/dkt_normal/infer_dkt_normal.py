from argparse import ArgumentParser
from pathlib import Path
import subprocess
from typing import List

import numpy as np
from dkt.pipelines.pipelines import DKTPipeline  # pyright: ignore[reportMissingImports]
from tools.common_utils import save_video  # pyright: ignore[reportMissingImports]


def _video_fps(video_path: Path) -> float:
    command: List[str] = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return 25.0

    raw = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not raw:
        return 25.0
    if "/" in raw:
        n, d = raw.split("/", 1)
        try:
            numerator = float(n)
            denominator = float(d)
        except ValueError:
            return 25.0
        if denominator == 0:
            return 25.0
        return numerator / denominator
    try:
        return float(raw)
    except ValueError:
        return 25.0


def _to_uint8_frames(frames: List[object]) -> np.ndarray:
    converted: List[np.ndarray] = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            array = frame
        else:
            array = np.array(frame)

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        converted.append(array)

    return np.stack(converted, axis=0)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--num-inference-steps", required=True, type=int)
    parser.add_argument("--window-size", required=True, type=int)
    parser.add_argument("--overlap", required=True, type=int)
    args = parser.parse_args()

    input_video = Path(args.input_video).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise RuntimeError(f"Input video not found: {input_video}")

    pipeline = DKTPipeline(model_path=args.model_id, is14B=True, is_depth=False)
    prediction = pipeline(
        str(input_video),
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        window_size=args.window_size,
        overlap=args.overlap,
        vis_pc=False,
        return_rgb=False,
    )

    colored_frames = prediction["colored_depth_map"]
    frames_u8 = _to_uint8_frames(colored_frames)

    stem = input_video.stem
    npz_path = output_dir / f"{stem}.npz"
    vis_path = output_dir / f"{stem}_vis.mp4"

    normals = (frames_u8.astype(np.float32) / 255.0) * 2.0 - 1.0
    np.savez_compressed(npz_path, normal=normals)

    save_video(list(frames_u8), str(vis_path), fps=_video_fps(input_video))


if __name__ == "__main__":
    main()
