import math
import os
import shlex
import shutil
import subprocess
import webbrowser
from time import sleep
from typing import Dict, List

from pipe.config import (
    CopyTracks,
    Encode,
    Ffmpeg,
    Interpolate,
    ManualDownload,
    Trim,
    Upscale,
)
from pydantic import BaseModel


class ExecutedStep(BaseModel):
    output_path: str
    extension: str


def execute_manual_download(
    step: ManualDownload, windows_downloads_dir: str, output_path_without_extension: str
) -> ExecutedStep:
    files_before = set(os.listdir(windows_downloads_dir))
    webbrowser.open(step.link)

    prev_sizes: Dict[str, int] = {}
    stable_counts: Dict[str, int] = {}
    detected_any = False

    while True:
        files_after = set(os.listdir(windows_downloads_dir))
        new_files = files_after - files_before

        if not new_files:
            sleep(0.5)
            continue

        detected_any = True

        current_sizes = {}
        for fname in list(new_files):
            full = os.path.join(windows_downloads_dir, fname)
            try:
                current_sizes[fname] = os.path.getsize(full)
            except OSError:
                continue

        tracked_files = set(prev_sizes.keys())
        for f in list(tracked_files):
            if f not in current_sizes:
                prev_sizes.pop(f, None)
                stable_counts.pop(f, None)

        for f, size in current_sizes.items():
            if f not in prev_sizes:
                prev_sizes[f] = size
                stable_counts[f] = 0
            else:
                if size == prev_sizes[f]:
                    stable_counts[f] = stable_counts.get(f, 0) + 1
                else:
                    stable_counts[f] = 0
                prev_sizes[f] = size

        stable_now = [
            f for f, c in stable_counts.items() if c >= 2 and f in current_sizes
        ]

        if (
            detected_any
            and new_files
            and all(f in stable_now for f in new_files if f in current_sizes)
        ):
            candidates = [
                f for f in new_files if f in current_sizes and f in stable_now
            ]
            if not candidates:
                sleep(0.5)
                continue
            chosen = max(
                candidates,
                key=lambda f: os.path.getmtime(os.path.join(windows_downloads_dir, f)),
            )
            new_file = chosen
            break

        sleep(0.5)

    extension = os.path.splitext(new_file)[1]

    os.makedirs(os.path.dirname(output_path_without_extension), exist_ok=True)
    shutil.move(
        os.path.join(windows_downloads_dir, new_file),
        (output_path := f"{output_path_without_extension}{extension}"),
    )
    return ExecutedStep(output_path=output_path, extension=extension)


PRORES_PROFILE_TO_FFMPEG = {
    "proxy": "0",
    "lt": "1",
    "422": "2",
    "hq": "3",
    "4444": "4",
    "4444xq": "5",
}


def get_ffmpeg_step_extension(step: Ffmpeg, previous_extension: str) -> str:
    for operation in step.operations:
        if isinstance(operation, Encode) and operation.codec == "prores":
            return ".mov"
    return previous_extension


def run_ffmpeg_step(input_path: str, step: Ffmpeg, output_path: str) -> None:
    if not step.operations:
        raise ValueError("ffmpeg step requires at least one operation.")

    trim_operation: Trim | None = None
    copy_tracks_operation: CopyTracks | None = None
    encode_operation: Encode | None = None

    for operation in step.operations:
        if isinstance(operation, Trim):
            if trim_operation is not None:
                raise ValueError("ffmpeg step supports at most one trim operation.")
            trim_operation = operation
            continue

        if isinstance(operation, CopyTracks):
            if copy_tracks_operation is not None:
                raise ValueError(
                    "ffmpeg step supports at most one copy_tracks operation."
                )
            copy_tracks_operation = operation
            continue

        if isinstance(operation, Encode):
            if encode_operation is not None:
                raise ValueError("ffmpeg step supports at most one encode operation.")
            encode_operation = operation
            continue

        raise ValueError(f"Unsupported ffmpeg operation: {operation.type}")

    if trim_operation is not None and trim_operation.end <= trim_operation.start:
        raise ValueError("ffmpeg trim operation requires end to be after start.")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video input not found: {input_path}")

    secondary_input_path: str | None = None
    if copy_tracks_operation is not None:
        secondary_input_path = copy_tracks_operation.source_path
        if not os.path.exists(secondary_input_path):
            raise FileNotFoundError(f"Source input not found: {secondary_input_path}")

    should_drop_subtitles = encode_operation is not None

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    primary_input_path = os.path.abspath(input_path)
    command: List[str] = ["ffmpeg", "-y"]

    if trim_operation is not None:
        command.extend(
            ["-ss", str(trim_operation.start), "-to", str(trim_operation.end)]
        )
    command.extend(["-i", primary_input_path])

    if secondary_input_path is not None:
        if trim_operation is not None:
            command.extend(
                ["-ss", str(trim_operation.start), "-to", str(trim_operation.end)]
            )
        command.extend(["-i", os.path.abspath(secondary_input_path)])

    if secondary_input_path is not None:
        command.extend(["-map", "0:v:0", "-map", "1:a?"])
        if not should_drop_subtitles:
            command.extend(["-map", "1:s?"])
        command.extend(["-map_metadata", "1", "-map_chapters", "1"])
        command.extend(["-c:a", "copy"])
        if not should_drop_subtitles:
            command.extend(["-c:s", "copy"])
    else:
        command.extend(["-map", "0:v", "-map", "0:a?"])
        if not should_drop_subtitles:
            command.extend(["-map", "0:s?"])
        command.extend(["-map_metadata", "0", "-map_chapters", "0"])
        if encode_operation is not None:
            command.extend(["-c:a", "copy"])
            if not should_drop_subtitles:
                command.extend(["-c:s", "copy"])
        else:
            command.extend(["-c", "copy"])

    if encode_operation is not None:
        if encode_operation.codec != "prores":
            raise ValueError(f"Unsupported encode codec: {encode_operation.codec}")
        command.extend(
            [
                "-c:v",
                "prores_ks",
                "-profile:v",
                PRORES_PROFILE_TO_FFMPEG[encode_operation.profile],
                "-pix_fmt",
                "yuv422p10le",
            ]
        )

    command.append(output_path)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "ffmpeg step failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc


def execute_ffmpeg(
    step: Ffmpeg, previous_step: ExecutedStep, output_path: str
) -> ExecutedStep:
    run_ffmpeg_step(
        input_path=previous_step.output_path,
        step=step,
        output_path=output_path,
    )
    return ExecutedStep(
        output_path=output_path,
        extension=get_ffmpeg_step_extension(step, previous_step.extension),
    )


def get_video_fps(input_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details: List[str] = [
            "Failed to read input video fps with ffprobe.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        if stderr:
            details.append(f"stderr:\n{stderr}")
        if stdout:
            details.append(f"stdout:\n{stdout}")
        raise RuntimeError("\n\n".join(details)) from exc

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("ffprobe returned empty output for input video fps.")

    try:
        raw = output.splitlines()[0].strip()
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            fps = float(numerator) / float(denominator)
        else:
            fps = float(raw)
        if fps <= 0:
            raise RuntimeError(
                f"Non-positive input fps parsed from ffprobe output: {raw}"
            )
        return fps
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse input video fps from ffprobe output: {output}"
        ) from exc


def get_video_width(input_path: str) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details: List[str] = [
            "Failed to read input video width with ffprobe.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        if stderr:
            details.append(f"stderr:\n{stderr}")
        if stdout:
            details.append(f"stdout:\n{stdout}")
        raise RuntimeError("\n\n".join(details)) from exc

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("ffprobe returned empty output for input video width.")

    try:
        width = int(output.splitlines()[0].strip())
        if width <= 0:
            raise RuntimeError(
                f"Non-positive input width parsed from ffprobe output: {width}"
            )
        return width
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse input video width from ffprobe output: {output}"
        ) from exc


def interpolate(input_path: str, output_path: str, fps: int) -> None:
    if fps <= 0:
        raise ValueError(
            f"Invalid interpolate fps={fps}. Fps must be a positive integer."
        )

    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    input_dir = os.path.dirname(input_abs)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    input_fps = get_video_fps(input_abs)
    if fps <= input_fps:
        raise ValueError(
            f"Requested fps={fps} is not larger than input fps={input_fps:.6f}. "
            "RIFE scale is derived from target/input fps and must be >= 2x."
        )

    required_scale = math.ceil(fps / input_fps)
    if required_scale < 2:
        required_scale = 2
    scale = 1
    while scale < required_scale:
        scale *= 2

    rife_image = os.environ.get("RIFE_IMAGE", "video-pipelines-rife:latest")
    docker_gpu_args = shlex.split(os.environ.get("DOCKER_GPU_ARGS", "--gpus all"))
    model_cache_dir = os.path.abspath(
        os.environ.get("RIFE_MODEL_CACHE_DIR", ".cache/rife-model")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "-v",
        f"{input_dir}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/opt/rife/train_log",
        rife_image,
        f"/io/in/{os.path.basename(input_abs)}",
        str(scale),
        f"/io/out/{os.path.basename(output_abs)}",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "RIFE docker interpolation failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
            f"Input fps: {input_fps:.6f}",
            f"Requested fps: {fps}",
            f"Computed RIFE scale: {scale}",
        ]
        details.append(
            "Hint: computed scale is the next power of two >= ceil(requested_fps / input_fps)."
        )
        raise RuntimeError("\n\n".join(details)) from exc


def execute_interpolate(
    step: Interpolate, previous_step: ExecutedStep, output_path: str
) -> ExecutedStep:
    interpolate(
        input_path=previous_step.output_path,
        output_path=output_path,
        fps=step.fps,
    )
    return ExecutedStep(output_path=output_path, extension=previous_step.extension)


def upscale(input_path: str, output_path: str, width: int) -> None:
    if width <= 0:
        raise ValueError(
            f"Invalid upscale width={width}. Width must be a positive integer."
        )

    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    input_dir = os.path.dirname(input_abs)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    input_width = get_video_width(input_abs)
    if width <= input_width:
        raise ValueError(
            f"Requested width={width} is not larger than input width={input_width}."
        )

    esrgan_image = os.environ.get("ESRGAN_IMAGE", "video-pipelines-esrgan:latest")
    docker_gpu_args = shlex.split(os.environ.get("DOCKER_GPU_ARGS", "--gpus all"))
    model_cache_dir = os.path.abspath(
        os.environ.get("ESRGAN_MODEL_CACHE_DIR", ".cache/esrgan-model")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "-v",
        f"{input_dir}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/opt/esrgan/models",
        esrgan_image,
        f"/io/in/{os.path.basename(input_abs)}",
        str(width),
        f"/io/out/{os.path.basename(output_abs)}",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "ESRGAN docker upscale failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
            f"Input width: {input_width}",
            f"Requested width: {width}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc


def execute_upscale(
    step: Upscale, previous_step: ExecutedStep, output_path: str
) -> ExecutedStep:
    upscale(
        input_path=previous_step.output_path,
        output_path=output_path,
        width=step.width,
    )
    return ExecutedStep(output_path=output_path, extension=previous_step.extension)
