import json
import math
import os
import shlex
import shutil
import subprocess
import webbrowser
from time import sleep
from typing import Any, Dict, List

from pipe.config import (
    Colmap,
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


def _run_subprocess(command: List[str], error_prefix: str) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            error_prefix,
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
    q_norm = math.sqrt((qw * qw) + (qx * qx) + (qy * qy) + (qz * qz))
    if q_norm == 0:
        raise ValueError("Invalid zero-norm quaternion in COLMAP output.")

    w = qw / q_norm
    x = qx / q_norm
    y = qy / q_norm
    z = qz / q_norm

    return [
        [
            1 - (2 * y * y) - (2 * z * z),
            (2 * x * y) - (2 * z * w),
            (2 * x * z) + (2 * y * w),
        ],
        [
            (2 * x * y) + (2 * z * w),
            1 - (2 * x * x) - (2 * z * z),
            (2 * y * z) - (2 * x * w),
        ],
        [
            (2 * x * z) - (2 * y * w),
            (2 * y * z) + (2 * x * w),
            1 - (2 * x * x) - (2 * y * y),
        ],
    ]


def _camera_center_from_qvec_tvec(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    tx: float,
    ty: float,
    tz: float,
) -> List[float]:
    rotation = _qvec_to_rotmat(qw, qx, qy, qz)
    return [
        -((rotation[0][0] * tx) + (rotation[1][0] * ty) + (rotation[2][0] * tz)),
        -((rotation[0][1] * tx) + (rotation[1][1] * ty) + (rotation[2][1] * tz)),
        -((rotation[0][2] * tx) + (rotation[1][2] * ty) + (rotation[2][2] * tz)),
    ]


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


def execute_colmap(
    step: Colmap,
    previous_step: ExecutedStep,
    output_path: str,
) -> ExecutedStep:
    try:
        import pycolmap  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "pycolmap is required for the colmap step. Install it with `uv sync` after adding the dependency."
        ) from exc

    input_video_path = os.path.abspath(previous_step.output_path)
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Video input not found: {input_video_path}")

    output_path_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path_abs)
    os.makedirs(output_dir, exist_ok=True)

    output_stem = os.path.splitext(os.path.basename(output_path_abs))[0]
    workspace_dir = os.path.join(output_dir, f"{output_stem}_colmap")
    images_dir = os.path.join(workspace_dir, "images")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    database_path = os.path.join(workspace_dir, "database.db")

    if os.path.isdir(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    frame_pattern = os.path.join(images_dir, f"frame_%06d.{step.image_format}")
    ffmpeg_command: List[str] = ["ffmpeg", "-y", "-i", input_video_path]
    if step.frame_rate is not None:
        ffmpeg_command.extend(["-vf", f"fps={step.frame_rate}"])
    if step.max_frames is not None:
        ffmpeg_command.extend(["-frames:v", str(step.max_frames)])
    ffmpeg_command.append(frame_pattern)

    _run_subprocess(ffmpeg_command, "Failed to extract frames for COLMAP step.")

    extracted_frames = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith(f".{step.image_format}")
        ]
    )
    if not extracted_frames:
        raise RuntimeError(
            "No frames were extracted for COLMAP. Check input video and frame extraction options."
        )

    sampling_fps = step.frame_rate
    if sampling_fps is None:
        sampling_fps = get_video_fps(input_video_path)

    pc: Any = pycolmap

    camera_mode = pc.CameraMode.SINGLE
    if step.camera_mode == "auto":
        camera_mode = pc.CameraMode.AUTO

    device = pc.Device.auto if step.use_gpu else pc.Device.cpu
    if (
        hasattr(pc, "FeatureExtractionOptions")
        and hasattr(pc, "FeatureMatchingOptions")
        and hasattr(pc, "SequentialPairingOptions")
    ):
        feature_extraction_options = pc.FeatureExtractionOptions()
        feature_extraction_options.max_image_size = step.max_image_size
        if hasattr(feature_extraction_options, "use_gpu"):
            feature_extraction_options.use_gpu = step.use_gpu

        pc.extract_features(
            database_path=database_path,
            image_path=images_dir,
            camera_mode=camera_mode,
            extraction_options=feature_extraction_options,
            device=device,
        )

        sequential_pairing_options = pc.SequentialPairingOptions()
        sequential_pairing_options.overlap = step.sequential_overlap
        sequential_pairing_options.loop_detection = step.loop_detection
        feature_matching_options = pc.FeatureMatchingOptions()
        if hasattr(feature_matching_options, "use_gpu"):
            feature_matching_options.use_gpu = step.use_gpu

        pc.match_sequential(
            database_path=database_path,
            matching_options=feature_matching_options,
            pairing_options=sequential_pairing_options,
            device=device,
        )
    else:
        sift_extraction_options = pc.SiftExtractionOptions()
        sift_extraction_options.max_image_size = step.max_image_size
        if hasattr(sift_extraction_options, "use_gpu"):
            sift_extraction_options.use_gpu = step.use_gpu

        pc.extract_features(
            database_path=database_path,
            image_path=images_dir,
            camera_mode=camera_mode,
            sift_options=sift_extraction_options,
            device=device,
        )

        sequential_matching_options = pc.SequentialMatchingOptions()
        sequential_matching_options.overlap = step.sequential_overlap
        sequential_matching_options.loop_detection = step.loop_detection
        sift_matching_options = pc.SiftMatchingOptions()
        if hasattr(sift_matching_options, "use_gpu"):
            sift_matching_options.use_gpu = step.use_gpu

        pc.match_sequential(
            database_path=database_path,
            sift_options=sift_matching_options,
            matching_options=sequential_matching_options,
            device=device,
        )

    use_global_mapper = (
        step.mapper == "glomap"
        and hasattr(pc, "global_mapping")
        and hasattr(pc, "GlobalPipelineOptions")
    )
    if not use_global_mapper:
        if step.mapper == "glomap":
            print(
                "Requested mapper='glomap', but installed pycolmap does not expose global mapping. Falling back to incremental mapping."
            )

        incremental_options = pc.IncrementalPipelineOptions()
        incremental_options.min_num_matches = step.min_num_matches
        incremental_options.max_num_models = step.max_num_models
        incremental_options.min_model_size = step.min_model_size
        if hasattr(incremental_options, "random_seed"):
            incremental_options.random_seed = step.random_seed
        if hasattr(incremental_options, "multiple_models"):
            incremental_options.multiple_models = step.max_num_models > 1

        reconstructions = pc.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=sparse_dir,
            options=incremental_options,
        )

        if not reconstructions:
            raise RuntimeError("COLMAP did not reconstruct any model.")

        best_model_idx, best_reconstruction = max(
            reconstructions.items(),
            key=lambda pair: (
                pair[1].num_reg_images(),
                pair[1].num_points3D(),
            ),
        )
    else:
        if hasattr(pc, "calibrate_view_graph"):
            pc.calibrate_view_graph(database_path=database_path)

        global_options = pc.GlobalPipelineOptions()
        global_options.mapper.global_positioning.use_gpu = step.use_gpu
        global_options.mapper.bundle_adjustment.ceres.use_gpu = step.use_gpu
        reconstructions = pc.global_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=sparse_dir,
            options=global_options,
        )

        if not reconstructions:
            raise RuntimeError("COLMAP global mapper did not reconstruct any model.")

        best_model_idx, best_reconstruction = max(
            reconstructions.items(),
            key=lambda pair: (
                pair[1].num_reg_images(),
                pair[1].num_points3D(),
            ),
        )

    best_model_dir = os.path.join(sparse_dir, str(best_model_idx))
    os.makedirs(best_model_dir, exist_ok=True)
    best_reconstruction.write_text(best_model_dir)
    best_reconstruction.write_text(sparse_dir)

    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")

    cameras: Dict[int, dict] = {}
    with open(cameras_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            camera_id = int(tokens[0])
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": tokens[1],
                "width": int(tokens[2]),
                "height": int(tokens[3]),
                "params": [float(value) for value in tokens[4:]],
            }

    trajectory_rows: List[dict] = []
    with open(images_path, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    for i in range(0, len(lines), 2):
        tokens = lines[i].split()
        image_id = int(tokens[0])
        qw = float(tokens[1])
        qx = float(tokens[2])
        qy = float(tokens[3])
        qz = float(tokens[4])
        tx = float(tokens[5])
        ty = float(tokens[6])
        tz = float(tokens[7])
        camera_id = int(tokens[8])
        image_name = tokens[9]

        frame_idx = None
        if image_name.startswith("frame_"):
            stem = image_name.split(".")[0]
            suffix = stem.replace("frame_", "")
            if suffix.isdigit():
                frame_idx = int(suffix)

        timestamp_seconds = None
        if frame_idx is not None and sampling_fps > 0:
            timestamp_seconds = (frame_idx - 1) / sampling_fps

        trajectory_rows.append(
            {
                "image_id": image_id,
                "image_name": image_name,
                "frame_index": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "camera_id": camera_id,
                "qvec_world_to_cam": [qw, qx, qy, qz],
                "tvec_world_to_cam": [tx, ty, tz],
                "camera_center_world": _camera_center_from_qvec_tvec(
                    qw=qw,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                ),
            }
        )

    trajectory_rows.sort(
        key=lambda row: (
            row["frame_index"] if row["frame_index"] is not None else 10**12,
            row["image_name"],
        )
    )

    mean_reprojection_error = None
    try:
        mean_reprojection_error = best_reconstruction.compute_mean_reprojection_error()
    except Exception:
        mean_reprojection_error = None

    output = {
        "input_video_path": input_video_path,
        "workspace_dir": workspace_dir,
        "images_dir": images_dir,
        "database_path": database_path,
        "sparse_models_dir": sparse_dir,
        "mapper": step.mapper,
        "best_model_index": best_model_idx,
        "best_model_dir": best_model_dir,
        "best_model_text_dir": sparse_dir,
        "sampling_fps": sampling_fps,
        "num_extracted_frames": len(extracted_frames),
        "num_reconstructed_models": len(reconstructions),
        "best_model_summary": {
            "num_images": best_reconstruction.num_images(),
            "num_registered_images": best_reconstruction.num_reg_images(),
            "num_points3D": best_reconstruction.num_points3D(),
            "mean_reprojection_error": mean_reprojection_error,
        },
        "cameras": [cameras[camera_id] for camera_id in sorted(cameras.keys())],
        "trajectory": trajectory_rows,
    }

    with open(output_path_abs, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    return ExecutedStep(output_path=output_path_abs, extension=".json")


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
