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
    Depth,
    DepthCrafterVariant,
    DepthProVariant,
    CopyTracks,
    DktNormalsVariant,
    DepthAnythingV2,
    DepthAnythingV2Variant,
    Encode,
    EsrganUpscaleVariant,
    Ffmpeg,
    Interpolate,
    ManualDownload,
    NormalCrafterVariant,
    Normals,
    SeedVR2UpscaleVariant,
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
    except FileNotFoundError as exc:
        missing_cmd_details: List[str] = [
            error_prefix,
            f"Command not found: {command[0]}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(missing_cmd_details)) from exc
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
    files_before_meta: Dict[str, tuple[int, int]] = {}
    for fname in files_before:
        full = os.path.join(windows_downloads_dir, fname)
        try:
            stat = os.stat(full)
        except OSError:
            continue
        files_before_meta[fname] = (stat.st_size, stat.st_mtime_ns)
    # Browser launch is opt-in; many WSL/headless environments have DISPLAY but no working URL opener.
    auto_open_links = os.environ.get("VIDEO_PIPELINES_AUTO_OPEN_LINKS", "0") == "1"
    has_desktop = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    opened = False
    if auto_open_links and has_desktop:
        try:
            opened = webbrowser.open(step.link)
        except Exception:
            opened = False

    if not opened:
        print("Could not auto-open browser. Download manually from:")
        print(step.link)
        print(f"Waiting for new or updated file in: {windows_downloads_dir}")

    prev_sizes: Dict[str, int] = {}
    stable_counts: Dict[str, int] = {}
    detected_any = False
    temp_suffixes = {".crdownload", ".part", ".tmp", ".download"}

    while True:
        files_after = set(os.listdir(windows_downloads_dir))
        changed_files: List[str] = []

        for fname in files_after:
            full = os.path.join(windows_downloads_dir, fname)
            extension = os.path.splitext(fname)[1].lower()
            if extension in temp_suffixes:
                continue

            try:
                stat = os.stat(full)
            except OSError:
                continue

            current_meta = (stat.st_size, stat.st_mtime_ns)
            previous_meta = files_before_meta.get(fname)
            if previous_meta is None or previous_meta != current_meta:
                changed_files.append(fname)

        if not changed_files:
            sleep(0.5)
            continue

        detected_any = True

        current_sizes = {}
        for fname in changed_files:
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
            and changed_files
            and all(f in stable_now for f in changed_files if f in current_sizes)
        ):
            candidates = [f for f in changed_files if f in current_sizes and f in stable_now]
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


def _read_colmap_dense_array(file_path: str) -> Any:
    import numpy as np  # pyright: ignore[reportMissingImports]

    with open(file_path, "rb") as f:
        header_parts: List[bytes] = []
        current = bytearray()
        while len(header_parts) < 3:
            char = f.read(1)
            if not char:
                raise RuntimeError(f"Invalid COLMAP dense array header: {file_path}")
            if char == b"&":
                header_parts.append(bytes(current))
                current.clear()
            else:
                current.extend(char)

        width = int(header_parts[0].decode("ascii"))
        height = int(header_parts[1].decode("ascii"))
        channels = int(header_parts[2].decode("ascii"))

        data = np.frombuffer(f.read(), dtype=np.float32)

    expected_size = width * height * channels
    if data.size != expected_size:
        raise RuntimeError(
            f"Unexpected COLMAP dense array size for {file_path}: "
            f"expected {expected_size} float32 values, got {data.size}"
        )

    if channels == 1:
        return data.reshape((height, width))

    return data.reshape((height, width, channels))


def _extract_depth_maps(
    pycolmap: Any,
    workspace_dir: str,
    sparse_dir: str,
    best_model_dir: str,
    images_dir: str,
    trajectory_rows: List[dict],
    use_gpu: bool,
) -> str:
    try:
        import numpy as np  # pyright: ignore[reportMissingImports]
        from PIL import Image  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "numpy and Pillow are required for depth extraction. Install with `uv sync`."
        ) from exc

    dense_dir = os.path.join(workspace_dir, "dense")
    os.makedirs(dense_dir, exist_ok=True)

    undistorted_dir = os.path.join(dense_dir, "undistorted")
    if os.path.isdir(undistorted_dir):
        shutil.rmtree(undistorted_dir)
    os.makedirs(undistorted_dir, exist_ok=True)

    undistort_errors: List[str] = []
    undistort_variants: List[dict] = [
        {
            "input_path": best_model_dir,
            "image_path": images_dir,
            "output_path": undistorted_dir,
            "output_type": "COLMAP",
        },
        {
            "input_path": sparse_dir,
            "image_path": images_dir,
            "output_path": undistorted_dir,
            "output_type": "COLMAP",
        },
    ]
    for kwargs in undistort_variants:
        try:
            pycolmap.undistort_images(**kwargs)
            undistort_errors = []
            break
        except Exception as exc:
            undistort_errors.append(str(exc))

    if undistort_errors:
        try:
            pycolmap.undistort_images(
                undistorted_dir,
                best_model_dir,
                images_dir,
            )
        except Exception as exc:
            undistort_errors.append(str(exc))
            raise RuntimeError(
                "Image undistortion failed using pycolmap API variants. "
                "Your pypi pycolmap build may not include dense reconstruction APIs. "
                f"Errors: {' | '.join(undistort_errors)}"
            ) from exc

    patch_errors: List[str] = []

    try:
        patch_options = pycolmap.PatchMatchOptions()
        if hasattr(patch_options, "geom_consistency"):
            patch_options.geom_consistency = True
        if hasattr(patch_options, "gpu_index"):
            patch_options.gpu_index = "0" if use_gpu else "-1"

        pycolmap.patch_match_stereo(
            workspace_path=undistorted_dir,
            workspace_format="COLMAP",
            options=patch_options,
        )
    except Exception as exc:
        patch_errors.append(str(exc))
        try:
            pycolmap.patch_match_stereo(
                workspace_path=undistorted_dir,
                workspace_format="COLMAP",
            )
        except Exception as exc2:
            patch_errors.append(str(exc2))
            if any(
                "requires CUDA" in error_text or "CUDA" in error_text
                for error_text in patch_errors
            ):
                raise RuntimeError(
                    "Stereo depth extraction requires CUDA in this pycolmap build, "
                    "but CUDA is not available in the current runtime."
                ) from exc2

            raise RuntimeError(
                "Stereo depth extraction failed with pycolmap API variants. "
                "Your pypi pycolmap build may not include dense reconstruction APIs. "
                f"Errors: {' | '.join(patch_errors)}"
            ) from exc2

    depth_output_dir = os.path.join(dense_dir, "depth_maps")
    os.makedirs(depth_output_dir, exist_ok=True)

    stereo_dir = os.path.join(undistorted_dir, "stereo")
    depth_maps_dir = os.path.join(stereo_dir, "depth_maps")

    if not os.path.exists(depth_maps_dir):
        raise RuntimeError(
            f"Depth maps directory not found: {depth_maps_dir}. "
            "PatchMatchStereo may have failed."
        )

    for row in trajectory_rows:
        image_name = row["image_name"]
        frame_idx = row["frame_index"]

        depth_file = os.path.splitext(image_name)[0] + ".geometric.bin"
        depth_path = os.path.join(depth_maps_dir, depth_file)

        if not os.path.exists(depth_path):
            continue

        try:
            if hasattr(pycolmap, "read_array"):
                depth_array = pycolmap.read_array(depth_path)
            else:
                depth_array = _read_colmap_dense_array(depth_path)

            depth_array = depth_array.astype(np.float32)

            valid_mask = depth_array > 0
            if valid_mask.any():
                depth_min = depth_array[valid_mask].min()
                depth_max = depth_array[valid_mask].max()
                if depth_max > depth_min:
                    normalized = (depth_array - depth_min) / (depth_max - depth_min)
                else:
                    normalized = np.zeros_like(depth_array)
            else:
                normalized = np.zeros_like(depth_array)

            depth_uint8 = (normalized * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_uint8, mode="L")

            output_name = (
                f"depth_{frame_idx:06d}.png"
                if frame_idx is not None
                else f"depth_{image_name.split('.')[0]}.png"
            )
            output_path = os.path.join(depth_output_dir, output_name)
            depth_img.save(output_path)
        except Exception:
            continue

    return depth_output_dir


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

    camera_mode = pycolmap.CameraMode.SINGLE
    if step.camera_mode == "auto":
        camera_mode = pycolmap.CameraMode.AUTO

    device = pycolmap.Device.auto if step.use_gpu else pycolmap.Device.cpu
    feature_extraction_options = pycolmap.FeatureExtractionOptions()
    feature_extraction_options.max_image_size = step.max_image_size
    feature_extraction_options.use_gpu = step.use_gpu

    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
        camera_mode=camera_mode,
        extraction_options=feature_extraction_options,
        device=device,
    )

    sequential_pairing_options = pycolmap.SequentialPairingOptions()
    sequential_pairing_options.overlap = step.sequential_overlap
    sequential_pairing_options.loop_detection = step.loop_detection
    feature_matching_options = pycolmap.FeatureMatchingOptions()
    feature_matching_options.use_gpu = step.use_gpu

    pycolmap.match_sequential(
        database_path=database_path,
        matching_options=feature_matching_options,
        pairing_options=sequential_pairing_options,
        device=device,
    )

    if step.mapper == "colmap":
        incremental_options = pycolmap.IncrementalPipelineOptions()
        incremental_options.min_num_matches = step.min_num_matches
        incremental_options.max_num_models = step.max_num_models
        incremental_options.min_model_size = step.min_model_size
        incremental_options.random_seed = step.random_seed
        incremental_options.multiple_models = step.max_num_models > 1

        reconstructions = pycolmap.incremental_mapping(
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
        global_options = pycolmap.GlobalPipelineOptions()
        global_options.mapper.global_positioning.use_gpu = step.use_gpu
        global_options.mapper.bundle_adjustment.ceres.use_gpu = step.use_gpu
        reconstructions = pycolmap.global_mapping(
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

    depth_dir = None
    depth_error: str | None = None
    if step.extract_depth:
        try:
            depth_dir = _extract_depth_maps(
                pycolmap=pycolmap,
                workspace_dir=workspace_dir,
                sparse_dir=sparse_dir,
                best_model_dir=best_model_dir,
                images_dir=images_dir,
                trajectory_rows=trajectory_rows,
                use_gpu=step.use_gpu,
            )
        except RuntimeError as exc:
            depth_error = str(exc)
            print(f"Depth extraction skipped: {depth_error}")

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
        "depth_dir": depth_dir,
        "depth_error": depth_error,
    }

    with open(output_path_abs, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    return ExecutedStep(output_path=output_path_abs, extension=".json")


def depth_anything_v2(
    input_path: str,
    output_path: str,
    encoder: str,
    input_size: int,
) -> None:
    if input_size <= 0:
        raise ValueError(
            f"Invalid input_size={input_size}. input_size must be a positive integer."
        )

    input_abs = os.path.abspath(input_path)
    if not os.path.exists(input_abs):
        raise FileNotFoundError(f"Video input not found: {input_abs}")

    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    output_stem = os.path.splitext(os.path.basename(output_abs))[0]
    depth_workspace_dir = os.path.join(output_dir, f"{output_stem}_depth_anything_v2")
    depth_workspace_name = os.path.basename(depth_workspace_dir)
    depth_maps_dir = os.path.join(depth_workspace_dir, "depth_maps")
    preview_video_path = os.path.join(depth_workspace_dir, "depth_preview.mp4")

    depth_image = os.environ.get(
        "DEPTH_ANYTHING_V2_IMAGE", "video-pipelines-depth-anything-v2:latest"
    )
    docker_gpu_args = shlex.split(os.environ.get("DOCKER_GPU_ARGS", "--gpus all"))
    model_cache_dir = os.path.abspath(
        os.environ.get("DEPTH_ANYTHING_V2_MODEL_CACHE_DIR", ".cache/depth-anything-v2")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    image_check = subprocess.run(
        ["docker", "image", "inspect", depth_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_check.returncode != 0:
        raise RuntimeError(
            "Depth Anything V2 docker image is not available locally: "
            f"{depth_image}\n\n"
            "Build it first with: make depth-anything-v2-image"
        )

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "-v",
        f"{os.path.dirname(input_abs)}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/opt/depth-anything-v2/checkpoints",
        depth_image,
        f"/io/in/{os.path.basename(input_abs)}",
        f"/io/out/{depth_workspace_name}",
        encoder,
        str(input_size),
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "Depth Anything V2 docker inference failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc

    depth_pngs = (
        [name for name in os.listdir(depth_maps_dir) if name.lower().endswith(".png")]
        if os.path.isdir(depth_maps_dir)
        else []
    )

    if not depth_pngs:
        raise RuntimeError(
            "Depth Anything V2 completed but produced no depth PNG files at "
            f"{depth_maps_dir}"
        )

    output = {
        "input_video_path": input_abs,
        "encoder": encoder,
        "input_size": input_size,
        "depth_workspace_dir": depth_workspace_dir,
        "depth_maps_dir": depth_maps_dir,
        "depth_preview_video_path": preview_video_path,
        "num_depth_maps": len(depth_pngs),
    }

    with open(output_abs, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


def execute_depth_anything_v2(
    step: DepthAnythingV2,
    previous_step: ExecutedStep,
    output_path: str,
) -> ExecutedStep:
    depth_anything_v2(
        input_path=previous_step.output_path,
        output_path=output_path,
        encoder=step.encoder,
        input_size=step.input_size,
    )
    return ExecutedStep(output_path=output_path, extension=".json")


def depth_crafter(
    input_path: str,
    output_path: str,
    max_res: int | None,
    process_length: int | None,
    target_fps: int | None,
) -> None:
    input_abs = os.path.abspath(input_path)
    if not os.path.exists(input_abs):
        raise FileNotFoundError(f"Video input not found: {input_abs}")

    if max_res is None or process_length is None or target_fps is None:
        probe_command: List[str] = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            input_abs,
        ]
        try:
            probe_result = subprocess.run(
                probe_command,
                check=True,
                capture_output=True,
                text=True,
            )
            probe_json = json.loads(probe_result.stdout)
            streams = probe_json["streams"]
            if not streams:
                raise ValueError("No video stream found in input file.")
            stream = streams[0]

            if max_res is None:
                width = int(stream["width"])
                height = int(stream["height"])
                max_res = max(width, height)

            if target_fps is None:
                avg_frame_rate = str(stream["avg_frame_rate"])
                if avg_frame_rate == "0/0":
                    raise ValueError("avg_frame_rate is 0/0.")
                numerator, denominator = avg_frame_rate.split("/", 1)
                fps = float(numerator) / float(denominator)
                target_fps = max(1, int(round(fps)))

            if process_length is None:
                nb_frames = stream.get("nb_frames")
                process_length = int(nb_frames) if nb_frames not in [None, "N/A"] else -1
        except Exception as exc:
            raise RuntimeError(
                "Failed to derive DepthCrafter defaults from input footage. "
                "Provide max_res/target_fps/process_length explicitly or ensure ffprobe metadata is available."
            ) from exc

    if max_res <= 0:
        raise ValueError(
            f"Invalid max_res={max_res}. max_res must be a positive integer."
        )

    if target_fps <= 0:
        raise ValueError(
            f"Invalid target_fps={target_fps}. target_fps must be a positive integer."
        )

    if process_length == 0 or process_length < -1:
        raise ValueError(
            f"Invalid process_length={process_length}. Use -1 for full clip length or a positive integer."
        )

    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    output_stem = os.path.splitext(os.path.basename(output_abs))[0]
    depth_workspace_dir = os.path.join(output_dir, f"{output_stem}_depth_crafter")
    depth_workspace_name = os.path.basename(depth_workspace_dir)

    input_stem = os.path.splitext(os.path.basename(input_abs))[0]
    depth_npz_path = os.path.join(depth_workspace_dir, f"{input_stem}.npz")
    preview_video_path = os.path.join(depth_workspace_dir, f"{input_stem}_vis.mp4")

    depth_image = os.environ.get(
        "DEPTH_CRAFTER_IMAGE", "video-pipelines-depth-crafter:latest"
    )
    docker_gpu_args = shlex.split(os.environ.get("DOCKER_GPU_ARGS", "--gpus all"))
    model_cache_dir = os.path.abspath(
        os.environ.get("DEPTH_CRAFTER_MODEL_CACHE_DIR", ".cache/depth-crafter")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    image_check = subprocess.run(
        ["docker", "image", "inspect", depth_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_check.returncode != 0:
        raise RuntimeError(
            "DepthCrafter docker image is not available locally: "
            f"{depth_image}\n\n"
            "Build it first with: make depth-crafter-image"
        )

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "-v",
        f"{os.path.dirname(input_abs)}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/root/.cache/huggingface",
        depth_image,
        f"/io/in/{os.path.basename(input_abs)}",
        f"/io/out/{depth_workspace_name}",
        str(max_res),
        str(process_length),
        str(target_fps),
    ]

    print(f"Running DepthCrafter docker command: {shlex.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "DepthCrafter docker inference failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc

    if not os.path.exists(depth_npz_path):
        raise RuntimeError(
            f"DepthCrafter completed but produced no depth NPZ file at {depth_npz_path}"
        )

    output = {
        "input_video_path": input_abs,
        "max_res": max_res,
        "process_length": process_length,
        "target_fps": target_fps,
        "depth_workspace_dir": depth_workspace_dir,
        "depth_npz_path": depth_npz_path,
        "depth_preview_video_path": preview_video_path,
    }

    with open(output_abs, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


def depth_pro(
    input_path: str,
    output_path: str,
    precision: str,
) -> None:
    if precision not in ["fp16", "fp32"]:
        raise ValueError(
            f"Unsupported precision={precision}. Use one of: fp16, fp32."
        )

    input_abs = os.path.abspath(input_path)
    if not os.path.exists(input_abs):
        raise FileNotFoundError(f"Video input not found: {input_abs}")

    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    output_stem = os.path.splitext(os.path.basename(output_abs))[0]
    depth_workspace_dir = os.path.join(output_dir, f"{output_stem}_depth_pro")
    depth_workspace_name = os.path.basename(depth_workspace_dir)
    depth_maps_dir = os.path.join(depth_workspace_dir, "depth_maps")
    preview_video_path = os.path.join(depth_workspace_dir, "depth_preview.mp4")
    metrics_npz_dir = os.path.join(depth_workspace_dir, "depth_npz")

    depth_image = os.environ.get("DEPTH_PRO_IMAGE", "video-pipelines-depth-pro:latest")
    docker_gpu_args_env = os.environ.get("DOCKER_GPU_ARGS")
    if docker_gpu_args_env is not None:
        docker_gpu_args = shlex.split(docker_gpu_args_env)
    else:
        nvidia_smi_check = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        docker_gpu_args = ["--gpus", "all"] if nvidia_smi_check.returncode == 0 else []
    model_cache_dir = os.path.abspath(
        os.environ.get("DEPTH_PRO_MODEL_CACHE_DIR", ".cache/depth-pro")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    image_check = subprocess.run(
        ["docker", "image", "inspect", depth_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_check.returncode != 0:
        raise RuntimeError(
            "Depth Pro docker image is not available locally: "
            f"{depth_image}\n\n"
            "Build it first with: make depth-pro-image"
        )

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "-v",
        f"{os.path.dirname(input_abs)}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/opt/ml-depth-pro/checkpoints",
        depth_image,
        f"/io/in/{os.path.basename(input_abs)}",
        f"/io/out/{depth_workspace_name}",
        precision,
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "Depth Pro docker inference failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc

    depth_pngs = (
        [name for name in os.listdir(depth_maps_dir) if name.lower().endswith(".png")]
        if os.path.isdir(depth_maps_dir)
        else []
    )
    depth_npzs = (
        [name for name in os.listdir(metrics_npz_dir) if name.lower().endswith(".npz")]
        if os.path.isdir(metrics_npz_dir)
        else []
    )

    if not depth_pngs:
        raise RuntimeError(
            "Depth Pro completed but produced no depth PNG files at "
            f"{depth_maps_dir}"
        )

    output = {
        "input_video_path": input_abs,
        "precision": precision,
        "depth_workspace_dir": depth_workspace_dir,
        "depth_maps_dir": depth_maps_dir,
        "depth_npz_dir": metrics_npz_dir,
        "depth_preview_video_path": preview_video_path,
        "num_depth_maps": len(depth_pngs),
        "num_depth_npz": len(depth_npzs),
    }

    with open(output_abs, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


def execute_depth(
    step: Depth,
    previous_step: ExecutedStep,
    output_path: str,
) -> ExecutedStep:
    variant = step.variant
    if isinstance(variant, DepthAnythingV2Variant):
        depth_anything_v2(
            input_path=previous_step.output_path,
            output_path=output_path,
            encoder=variant.encoder,
            input_size=variant.input_size,
        )
    elif isinstance(variant, DepthCrafterVariant):
        depth_crafter(
            input_path=previous_step.output_path,
            output_path=output_path,
            max_res=variant.max_res,
            process_length=variant.process_length,
            target_fps=variant.target_fps,
        )
    elif isinstance(variant, DepthProVariant):
        depth_pro(
            input_path=previous_step.output_path,
            output_path=output_path,
            precision=variant.precision,
        )
    else:
        raise ValueError(f"Unsupported depth variant: {variant.type}")

    return ExecutedStep(output_path=output_path, extension=".json")


def normal_crafter(
    input_path: str,
    output_path: str,
    cpu_offload: str,
    unet_path: str,
    pre_train_path: str,
    max_res: int | None,
    process_length: int | None,
    target_fps: int | None,
    window_size: int,
    time_step_size: int,
    decode_chunk_size: int,
) -> None:
    input_abs = os.path.abspath(input_path)
    if not os.path.exists(input_abs):
        raise FileNotFoundError(f"Video input not found: {input_abs}")

    if max_res is None or process_length is None or target_fps is None:
        probe_command: List[str] = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            input_abs,
        ]
        try:
            probe_result = subprocess.run(
                probe_command,
                check=True,
                capture_output=True,
                text=True,
            )
            probe_json = json.loads(probe_result.stdout)
            streams = probe_json["streams"]
            if not streams:
                raise ValueError("No video stream found in input file.")
            stream = streams[0]

            if max_res is None:
                width = int(stream["width"])
                height = int(stream["height"])
                max_res = max(width, height)

            if target_fps is None:
                avg_frame_rate = str(stream["avg_frame_rate"])
                if avg_frame_rate == "0/0":
                    raise ValueError("avg_frame_rate is 0/0.")
                numerator, denominator = avg_frame_rate.split("/", 1)
                fps = float(numerator) / float(denominator)
                target_fps = max(1, int(round(fps)))

            if process_length is None:
                nb_frames = stream.get("nb_frames")
                process_length = (
                    int(nb_frames) if nb_frames not in [None, "N/A"] else -1
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to derive NormalCrafter defaults from input footage. "
                "Provide max_res/target_fps/process_length explicitly or ensure ffprobe metadata is available."
            ) from exc

    if max_res <= 0:
        raise ValueError(
            f"Invalid max_res={max_res}. max_res must be a positive integer."
        )

    if target_fps <= 0:
        raise ValueError(
            f"Invalid target_fps={target_fps}. target_fps must be a positive integer."
        )

    if process_length == 0 or process_length < -1:
        raise ValueError(
            f"Invalid process_length={process_length}. Use -1 for full clip length or a positive integer."
        )

    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    output_stem = os.path.splitext(os.path.basename(output_abs))[0]
    normals_workspace_dir = os.path.join(output_dir, f"{output_stem}_normal_crafter")
    normals_workspace_name = os.path.basename(normals_workspace_dir)

    input_stem = os.path.splitext(os.path.basename(input_abs))[0]
    normals_npz_path = os.path.join(normals_workspace_dir, f"{input_stem}.npz")
    preview_video_path = os.path.join(normals_workspace_dir, f"{input_stem}_vis.mp4")

    normals_image = os.environ.get(
        "NORMAL_CRAFTER_IMAGE", "video-pipelines-normal-crafter:latest"
    )
    docker_gpu_args = shlex.split(os.environ.get("DOCKER_GPU_ARGS", "--gpus all"))
    model_cache_dir = os.path.abspath(
        os.environ.get("NORMAL_CRAFTER_MODEL_CACHE_DIR", ".cache/normal-crafter")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    image_check = subprocess.run(
        ["docker", "image", "inspect", normals_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_check.returncode != 0:
        raise RuntimeError(
            "NormalCrafter docker image is not available locally: "
            f"{normals_image}\n\n"
            "Build it first with: make normal-crafter-image"
        )

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "--entrypoint",
        "python3",
        "-v",
        f"{os.path.dirname(input_abs)}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/root/.cache/huggingface",
        normals_image,
        "/opt/NormalCrafter/run.py",
        "--video-path",
        f"/io/in/{os.path.basename(input_abs)}",
        "--save-folder",
        f"/io/out/{normals_workspace_name}",
        "--unet-path",
        unet_path,
        "--pre-train-path",
        pre_train_path,
        "--cpu-offload",
        cpu_offload,
        "--max-res",
        str(max_res),
        "--process-length",
        str(process_length),
        "--target-fps",
        str(target_fps),
        "--window-size",
        str(window_size),
        "--time-step-size",
        str(time_step_size),
        "--save-npz=True",
    ]

    help_command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "--entrypoint",
        "python3",
        normals_image,
        "/opt/NormalCrafter/run.py",
        "--help",
    ]
    decode_chunk_supported = False
    try:
        help_result = subprocess.run(
            help_command,
            check=True,
            capture_output=True,
            text=True,
        )
        decode_chunk_supported = "--decode-chunk-size" in (
            (help_result.stdout or "") + (help_result.stderr or "")
        )
    except subprocess.CalledProcessError:
        decode_chunk_supported = False

    if decode_chunk_supported:
        command.extend(["--decode-chunk-size", str(decode_chunk_size)])
    else:
        print(
            "NormalCrafter run.py does not support --decode-chunk-size; "
            "continuing without it."
        )

    print(f"Running NormalCrafter docker command: {shlex.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_tail: List[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output_tail.append(line.rstrip("\n"))
            if len(output_tail) > 120:
                output_tail.pop(0)

        exit_code = process.wait()
        if exit_code == 0:
            output_tail = []
        if exit_code != 0:
            details: List[str] = [
                "NormalCrafter docker inference failed.",
                f"Exit code: {exit_code}",
                f"Command: {shlex.join(command)}",
            ]
            if exit_code == 137:
                details.append(
                    "Likely cause: process was killed due to memory pressure (OOM). "
                    "Try lowering NormalCrafter settings: max_res, target_fps, "
                    "window_size, time_step_size, or keep decode_chunk_size at 1."
                )
            if output_tail:
                details.append("output (tail):\n" + "\n".join(output_tail))
            raise RuntimeError("\n\n".join(details))
    except OSError as exc:
        os_error_details: List[str] = [
            "NormalCrafter docker inference failed.",
            f"OSError: {exc}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(os_error_details)) from exc

    if not os.path.exists(normals_npz_path):
        raise RuntimeError(
            "NormalCrafter completed but produced no normals NPZ file at "
            f"{normals_npz_path}"
        )

    output = {
        "input_video_path": input_abs,
        "cpu_offload": cpu_offload,
        "unet_path": unet_path,
        "pre_train_path": pre_train_path,
        "max_res": max_res,
        "process_length": process_length,
        "target_fps": target_fps,
        "window_size": window_size,
        "time_step_size": time_step_size,
        "decode_chunk_size": decode_chunk_size,
        "normals_workspace_dir": normals_workspace_dir,
        "normals_npz_path": normals_npz_path,
        "normals_preview_video_path": preview_video_path,
    }

    with open(output_abs, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


def execute_normals(
    step: Normals,
    previous_step: ExecutedStep,
    output_path: str,
) -> ExecutedStep:
    variant = step.variant
    if isinstance(variant, NormalCrafterVariant):
        normal_crafter(
            input_path=previous_step.output_path,
            output_path=output_path,
            cpu_offload=variant.cpu_offload,
            unet_path=variant.unet_path,
            pre_train_path=variant.pre_train_path,
            max_res=variant.max_res,
            process_length=variant.process_length,
            target_fps=variant.target_fps,
            window_size=variant.window_size,
            time_step_size=variant.time_step_size,
            decode_chunk_size=variant.decode_chunk_size,
        )
    elif isinstance(variant, DktNormalsVariant):
        dkt_normals(
            input_path=previous_step.output_path,
            output_path=output_path,
            model_id=variant.model_id,
            height=variant.height,
            width=variant.width,
            num_inference_steps=variant.num_inference_steps,
            window_size=variant.window_size,
            overlap=variant.overlap,
        )
    else:
        raise ValueError(f"Unsupported normals variant: {variant.type}")

    return ExecutedStep(output_path=output_path, extension=".json")


def dkt_normals(
    input_path: str,
    output_path: str,
    model_id: str,
    height: int,
    width: int,
    num_inference_steps: int,
    window_size: int,
    overlap: int,
) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Invalid size ({height}x{width}). Height and width must be positive integers."
        )
    if num_inference_steps <= 0:
        raise ValueError(
            f"Invalid num_inference_steps={num_inference_steps}. Must be a positive integer."
        )
    if window_size <= 0:
        raise ValueError(
            f"Invalid window_size={window_size}. Must be a positive integer."
        )
    if overlap < 0:
        raise ValueError(f"Invalid overlap={overlap}. Must be non-negative.")
    if overlap >= window_size:
        raise ValueError(
            f"Invalid overlap={overlap}. overlap must be less than window_size={window_size}."
        )

    input_abs = os.path.abspath(input_path)
    if not os.path.exists(input_abs):
        raise FileNotFoundError(f"Video input not found: {input_abs}")

    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    output_stem = os.path.splitext(os.path.basename(output_abs))[0]
    normals_workspace_dir = os.path.join(output_dir, f"{output_stem}_dkt")
    normals_workspace_name = os.path.basename(normals_workspace_dir)

    input_stem = os.path.splitext(os.path.basename(input_abs))[0]
    normals_npz_path = os.path.join(normals_workspace_dir, f"{input_stem}.npz")
    preview_video_path = os.path.join(normals_workspace_dir, f"{input_stem}_vis.mp4")

    dkt_image = os.environ.get("DKT_NORMAL_IMAGE", "video-pipelines-dkt-normal:latest")
    docker_gpu_args = shlex.split(os.environ.get("DOCKER_GPU_ARGS", "--gpus all"))
    model_cache_dir = os.path.abspath(
        os.environ.get("DKT_NORMAL_MODEL_CACHE_DIR", ".cache/dkt-normal-model")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    image_check = subprocess.run(
        ["docker", "image", "inspect", dkt_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_check.returncode != 0:
        raise RuntimeError(
            "DKT docker image is not available locally: "
            f"{dkt_image}\n\n"
            "Build it first with: make dkt-normal-image"
        )

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "-v",
        f"{os.path.dirname(input_abs)}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/opt/DKT/checkpoints",
        dkt_image,
        f"/io/in/{os.path.basename(input_abs)}",
        f"/io/out/{normals_workspace_name}",
        model_id,
        str(height),
        str(width),
        str(num_inference_steps),
        str(window_size),
        str(overlap),
    ]
    print(f"Running DKT normals docker command: {shlex.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "DKT normals docker inference failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc

    if not os.path.exists(normals_npz_path):
        raise RuntimeError(
            f"DKT normals completed but produced no NPZ file at {normals_npz_path}"
        )

    output = {
        "input_video_path": input_abs,
        "model_id": model_id,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "window_size": window_size,
        "overlap": overlap,
        "normals_workspace_dir": normals_workspace_dir,
        "normals_npz_path": normals_npz_path,
        "normals_preview_video_path": preview_video_path,
    }

    with open(output_abs, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


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


def upscale_esrgan(input_path: str, output_path: str, width: int) -> None:
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


def _get_video_dimensions(input_path: str) -> tuple[int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        input_path,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to probe input dimensions with ffprobe. "
            f"Command: {shlex.join(command)}"
        ) from exc

    output = result.stdout.strip()
    if "x" not in output:
        raise RuntimeError(f"Unexpected ffprobe width/height output: {output}")

    width_raw, height_raw = output.split("x", 1)
    try:
        width = int(width_raw)
        height = int(height_raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse input dimensions from ffprobe output: {output}"
        ) from exc

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Non-positive video dimensions from ffprobe: {width}x{height}")

    return width, height


def _seedvr2_preflight_import_check(seedvr2_image: str, docker_gpu_args: List[str]) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "--entrypoint",
        "python3",
        seedvr2_image,
        "-c",
        "from diffusers.loaders import single_file_model; print('seedvr2_preflight=ok')",
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return

    stderr_text = result.stderr.strip()
    stdout_text = result.stdout.strip()
    details: List[str] = [
        "SeedVR2 preflight check failed before inference.",
        "The SeedVR2 image cannot import diffusers single_file loader.",
        f"Image: {seedvr2_image}",
        f"Exit code: {result.returncode}",
        f"Command: {shlex.join(command)}",
        "If you recently changed Dockerfile, rebuild with: make seedvr2-image",
    ]
    if stdout_text:
        details.append(f"stdout:\n{stdout_text}")
    if stderr_text:
        details.append(f"stderr:\n{stderr_text}")
    raise RuntimeError("\n\n".join(details))


def upscale_seedvr2(
    input_path: str,
    output_path: str,
    variant: SeedVR2UpscaleVariant,
) -> None:
    if variant.width <= 0:
        raise ValueError(
            f"Invalid upscale width={variant.width}. Width must be a positive integer."
        )

    if variant.batch_size <= 0:
        raise ValueError(
            f"Invalid SeedVR2 batch_size={variant.batch_size}. Must be positive."
        )
    if variant.batch_size != 1 and ((variant.batch_size - 1) % 4 != 0):
        raise ValueError(
            "SeedVR2 batch_size must follow 4n+1 (1, 5, 9, ...). "
            f"Got: {variant.batch_size}."
        )
    if variant.blocks_to_swap < 0:
        raise ValueError(
            f"Invalid SeedVR2 blocks_to_swap={variant.blocks_to_swap}. Must be non-negative."
        )
    if variant.temporal_overlap < 0:
        raise ValueError(
            f"Invalid SeedVR2 temporal_overlap={variant.temporal_overlap}. Must be non-negative."
        )

    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    input_dir = os.path.dirname(input_abs)
    output_dir = os.path.dirname(output_abs)
    os.makedirs(output_dir, exist_ok=True)

    input_width, input_height = _get_video_dimensions(input_abs)
    if variant.width <= input_width:
        raise ValueError(
            f"Requested width={variant.width} is not larger than input width={input_width}."
        )

    scale = variant.width / input_width
    input_short_side = min(input_width, input_height)
    target_short_side = max(2, int(round(input_short_side * scale)))

    seedvr2_image = os.environ.get("SEEDVR2_IMAGE", "video-pipelines-seedvr2:latest")
    docker_gpu_args = shlex.split(os.environ.get("DOCKER_GPU_ARGS", "--gpus all"))
    model_cache_dir = os.path.abspath(
        os.environ.get("SEEDVR2_MODEL_CACHE_DIR", ".cache/seedvr2-model")
    )
    os.makedirs(model_cache_dir, exist_ok=True)

    image_check = subprocess.run(
        ["docker", "image", "inspect", seedvr2_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_check.returncode != 0:
        raise RuntimeError(
            "SeedVR2 docker image is not available locally: "
            f"{seedvr2_image}\n\n"
            "Build it first with: make seedvr2-image"
        )

    _seedvr2_preflight_import_check(
        seedvr2_image=seedvr2_image,
        docker_gpu_args=docker_gpu_args,
    )

    command: List[str] = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args,
        "--entrypoint",
        "python3",
        "-v",
        f"{input_dir}:/io/in:ro",
        "-v",
        f"{output_dir}:/io/out",
        "-v",
        f"{model_cache_dir}:/opt/seedvr2/models/SEEDVR2",
        seedvr2_image,
        "/opt/seedvr2/inference_cli.py",
        f"/io/in/{os.path.basename(input_abs)}",
        "--output",
        f"/io/out/{os.path.basename(output_abs)}",
        "--model_dir",
        "/opt/seedvr2/models/SEEDVR2",
        "--dit_model",
        variant.model,
        "--resolution",
        str(target_short_side),
        "--max_resolution",
        str(variant.max_resolution),
        "--batch_size",
        str(variant.batch_size),
        "--temporal_overlap",
        str(variant.temporal_overlap),
        "--blocks_to_swap",
        str(variant.blocks_to_swap),
        "--dit_offload_device",
        variant.dit_offload_device,
        "--vae_offload_device",
        variant.vae_offload_device,
        "--tensor_offload_device",
        variant.tensor_offload_device,
        "--vae_encode_tile_size",
        str(variant.vae_encode_tile_size),
        "--vae_encode_tile_overlap",
        str(variant.vae_encode_tile_overlap),
        "--vae_decode_tile_size",
        str(variant.vae_decode_tile_size),
        "--vae_decode_tile_overlap",
        str(variant.vae_decode_tile_overlap),
        "--video_backend",
        "ffmpeg",
    ]

    if variant.swap_io_components:
        command.append("--swap_io_components")
    if variant.vae_encode_tiled:
        command.append("--vae_encode_tiled")
    if variant.vae_decode_tiled:
        command.append("--vae_decode_tiled")
    if variant.cache_dit:
        command.append("--cache_dit")
    if variant.cache_vae:
        command.append("--cache_vae")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "SeedVR2 docker upscale failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
            f"Input dimensions: {input_width}x{input_height}",
            f"Requested output width: {variant.width}",
            f"Derived SeedVR2 short-side resolution: {target_short_side}",
        ]
        details.append(
            "OOM tip: lower batch_size (keep 4n+1), increase blocks_to_swap, "
            "and reduce max_resolution/tile sizes."
        )
        if exc.stdout:
            details.append(f"stdout:\n{exc.stdout.strip()}")
        if exc.stderr:
            details.append(f"stderr:\n{exc.stderr.strip()}")
        raise RuntimeError("\n\n".join(details)) from exc


def execute_upscale(
    step: Upscale, previous_step: ExecutedStep, output_path: str
) -> ExecutedStep:
    variant = step.variant
    if isinstance(variant, EsrganUpscaleVariant):
        upscale_esrgan(
            input_path=previous_step.output_path,
            output_path=output_path,
            width=variant.width,
        )
    elif isinstance(variant, SeedVR2UpscaleVariant):
        upscale_seedvr2(
            input_path=previous_step.output_path,
            output_path=output_path,
            variant=variant,
        )
    else:
        raise ValueError(f"Unsupported upscale variant: {variant.type}")

    return ExecutedStep(output_path=output_path, extension=previous_step.extension)
