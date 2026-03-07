import glob
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Dict, List, Literal, Optional

import duckdb
import ffmpeg
import jinja2
import webbrowser
import yaml
from pipe.chart import Chart
from pipe.config import (
    Config,
    CopyTracks,
    Episode,
    Interpolate,
    ManualDownload,
    Path,
    Trim,
    Upscale,
)
from pydantic import BaseModel


class ExecutedStep(BaseModel):
    output_path: str
    extension: str


class LoadedConfig(BaseModel):
    config: Config
    chart_path: str
    chart_json: str
    rendered_config_json: str


StepEventType = Literal["start", "completed", "skipped", "failed"]
STEP_EVENTS_TABLE = "step_events_v3"


class MediaFileInfo(BaseModel):
    path: str
    extension: str
    size_bytes: Optional[int]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    duration_ms: Optional[int]
    frame_count: Optional[int]
    sha256: Optional[str]
    metadata_text: Optional[str]


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _parse_fraction_to_float(value: Optional[str]) -> Optional[float]:
    if not value or value in ["0/0", "N/A"]:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
            if den == 0:
                return None
            return num / den
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_media_file_info(file_path: str, include_hash: bool = False) -> MediaFileInfo:
    file_abs = os.path.abspath(file_path)
    extension = os.path.splitext(file_abs)[1]

    if not os.path.exists(file_abs):
        return MediaFileInfo(
            path=file_abs,
            extension=extension,
            size_bytes=None,
            width=None,
            height=None,
            fps=None,
            duration_ms=None,
            frame_count=None,
            sha256=None,
            metadata_text=None,
        )

    size_bytes = os.path.getsize(file_abs)
    try:
        probe = ffmpeg.probe(file_abs)
        metadata_text = json.dumps(probe, sort_keys=True)
    except ffmpeg.Error as exc:
        stderr_text = ""
        if exc.stderr:
            try:
                stderr_text = exc.stderr.decode("utf-8", errors="replace")
            except Exception:
                stderr_text = str(exc.stderr)
        sha256 = hash_file_sha256(file_abs) if include_hash else None
        return MediaFileInfo(
            path=file_abs,
            extension=extension,
            size_bytes=size_bytes,
            width=None,
            height=None,
            fps=None,
            duration_ms=None,
            frame_count=None,
            sha256=sha256,
            metadata_text=stderr_text or str(exc),
        )

    format_info = probe.get("format", {})
    streams = probe.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )

    width = _safe_int(video_stream.get("width") if video_stream else None)
    height = _safe_int(video_stream.get("height") if video_stream else None)

    fps = None
    if video_stream:
        fps = _parse_fraction_to_float(video_stream.get("avg_frame_rate"))
        if fps is None:
            fps = _parse_fraction_to_float(video_stream.get("r_frame_rate"))

    duration_seconds: Optional[float] = None
    try:
        if "duration" in format_info:
            duration_seconds = float(format_info["duration"])
    except (TypeError, ValueError):
        duration_seconds = None

    if duration_seconds is None and video_stream:
        try:
            if "duration" in video_stream:
                duration_seconds = float(video_stream["duration"])
        except (TypeError, ValueError):
            duration_seconds = None

    duration_ms = int(duration_seconds * 1000) if duration_seconds is not None else None

    frame_count = _safe_int(video_stream.get("nb_frames") if video_stream else None)
    if frame_count is None and fps is not None and duration_seconds is not None:
        frame_count = int(round(fps * duration_seconds))

    sha256 = hash_file_sha256(file_abs) if include_hash else None
    return MediaFileInfo(
        path=file_abs,
        extension=extension,
        size_bytes=size_bytes,
        width=width,
        height=height,
        fps=fps,
        duration_ms=duration_ms,
        frame_count=frame_count,
        sha256=sha256,
        metadata_text=metadata_text,
    )


class StepEventLogger:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {STEP_EVENTS_TABLE} (
                event_timestamp TIMESTAMP,
                job STRUCT(
                    name VARCHAR,
                    chart_path VARCHAR,
                    chart_json VARCHAR,
                    rendered_config_json VARCHAR,
                    pipeline_index INTEGER,
                    n_pipelines INTEGER,
                    start_timestamp TIMESTAMP,
                    end_timestamp TIMESTAMP
                ),
                pipeline STRUCT(
                    name VARCHAR,
                    metadata_json VARCHAR,
                    input_json VARCHAR,
                    total_n_steps INTEGER,
                    start_timestamp TIMESTAMP,
                    end_timestamp TIMESTAMP
                ),
                step STRUCT(
                    index INTEGER,
                    type VARCHAR,
                    event VARCHAR,
                    step_json VARCHAR,
                    error_message VARCHAR
                ),
                file STRUCT(
                    path VARCHAR,
                    extension VARCHAR,
                    size_bytes BIGINT,
                    width INTEGER,
                    height INTEGER,
                    fps DOUBLE,
                    duration_ms BIGINT,
                    frame_count BIGINT,
                    sha256 VARCHAR
                ),
                raw_media_metadata TEXT
            )
            """
        )

    def log_step_event(
        self,
        *,
        event_timestamp: datetime,
        chart_path: str,
        chart_json: str,
        rendered_config_json: str,
        job_name: str,
        pipeline_index: int,
        n_pipelines: int,
        job_end_timestamp: Optional[datetime],
        pipeline_name: str,
        pipeline_metadata_json: str,
        pipeline_input_json: str,
        total_n_steps: int,
        pipeline_start_timestamp: datetime,
        pipeline_end_timestamp: Optional[datetime],
        step_index: int,
        step_type: str,
        step_json: str,
        step_event: StepEventType,
        job_start_timestamp: datetime,
        file_info: MediaFileInfo,
        error_message: Optional[str],
    ) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {STEP_EVENTS_TABLE} (
                event_timestamp,
                job,
                pipeline,
                step,
                file,
                raw_media_metadata
            )
            SELECT
                ?,
                struct_pack(
                    name := ?,
                    chart_path := ?,
                    chart_json := ?,
                    rendered_config_json := ?,
                    pipeline_index := ?,
                    n_pipelines := ?,
                    start_timestamp := ?,
                    end_timestamp := ?
                ),
                struct_pack(
                    name := ?,
                    metadata_json := ?,
                    input_json := ?,
                    total_n_steps := ?,
                    start_timestamp := ?,
                    end_timestamp := ?
                ),
                struct_pack(
                    index := ?,
                    type := ?,
                    event := ?,
                    step_json := ?,
                    error_message := ?
                ),
                struct_pack(
                    path := ?,
                    extension := ?,
                    size_bytes := ?,
                    width := ?,
                    height := ?,
                    fps := ?,
                    duration_ms := ?,
                    frame_count := ?,
                    sha256 := ?
                ),
                ?
            """,
            [
                event_timestamp,
                job_name,
                chart_path,
                chart_json,
                rendered_config_json,
                pipeline_index,
                n_pipelines,
                job_start_timestamp,
                job_end_timestamp,
                pipeline_name,
                pipeline_metadata_json,
                pipeline_input_json,
                total_n_steps,
                pipeline_start_timestamp,
                pipeline_end_timestamp,
                step_index,
                step_type,
                step_event,
                step_json,
                error_message,
                file_info.path,
                file_info.extension,
                file_info.size_bytes,
                file_info.width,
                file_info.height,
                file_info.fps,
                file_info.duration_ms,
                file_info.frame_count,
                file_info.sha256,
                file_info.metadata_text,
            ],
        )

    def should_skip_step(
        self,
        *,
        job_name: str,
        pipeline_name: str,
        step_index: int,
        step_json: str,
        output_file_hash: str,
    ) -> bool:
        row = self._conn.execute(
            f"""
            SELECT 1
            FROM {STEP_EVENTS_TABLE}
            WHERE step.event = 'completed'
              AND job.name = ?
              AND pipeline.name = ?
              AND step.index = ?
              AND step.step_json = ?
              AND file.sha256 = ?
            LIMIT 1
            """,
            [job_name, pipeline_name, step_index, step_json, output_file_hash],
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self._conn.close()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_file_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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

        # Gather current sizes for currently present new files
        current_sizes = {}
        for fname in list(new_files):
            full = os.path.join(windows_downloads_dir, fname)
            try:
                current_sizes[fname] = os.path.getsize(full)
            except OSError:
                # file may have disappeared between listing and size check
                continue

        # Update stability tracking
        # reset tracking for files that disappeared
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

        # Consider a file stable if its size hasn't changed for 2 consecutive checks
        stable_now = [
            f for f, c in stable_counts.items() if c >= 2 and f in current_sizes
        ]

        # Only finish when we've seen at least one new file and all currently present new files are stable
        # (this handles part -> final replacement because set(new_files) will change and we continue until stable)
        if (
            detected_any
            and new_files
            and all(f in stable_now for f in new_files if f in current_sizes)
        ):
            # choose the stable file with latest mtime (final file if a part was replaced)
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


def trim(input_path: str, output_path: str, start: timedelta, end: timedelta) -> None:
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-ss",
        str(start),
        "-to",
        str(end),
        "-c",
        "copy",
        output_path,
        "-y",
    ]
    subprocess.run(command, check=True)


def execute_trim(
    step: Trim, previous_step: ExecutedStep, output_path: str
) -> ExecutedStep:
    trim(
        input_path=previous_step.output_path,
        output_path=output_path,
        start=step.start,
        end=step.end,
    )
    return ExecutedStep(output_path=output_path, extension=previous_step.extension)


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


def copy_tracks(video_input_path: str, source_path: str, output_path: str) -> None:
    if not os.path.exists(video_input_path):
        raise FileNotFoundError(f"Video input not found: {video_input_path}")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source input not found: {source_path}")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    command: List[str] = [
        "ffmpeg",
        "-i",
        video_input_path,
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
        "-c",
        "copy",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-y",
        output_path,
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "Copy tracks step failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {shlex.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc


def execute_copy_tracks(
    step: CopyTracks, previous_step: ExecutedStep, output_path: str
) -> ExecutedStep:
    copy_tracks(
        video_input_path=previous_step.output_path,
        source_path=step.source_path,
        output_path=output_path,
    )
    return ExecutedStep(output_path=output_path, extension=previous_step.extension)


def get_pipeline_artifact_dir(config: Config, pipeline_name: str) -> str:
    return os.path.join(config.artifact_dir, config.job.name, pipeline_name)


def resolve_initial_step(
    config: Config,
    pipeline_name: str,
    input_step: Path | ManualDownload | Episode,
) -> ExecutedStep:
    if isinstance(input_step, ManualDownload):
        output_path_without_extension = os.path.join(
            get_pipeline_artifact_dir(config, pipeline_name),
            input_step.__class__.__name__,
        )
        matches = glob.glob(f"{output_path_without_extension}*")
        if matches and not config.full_refresh:
            chosen = max(matches, key=os.path.getmtime)
            return ExecutedStep(
                output_path=chosen,
                extension=os.path.splitext(chosen)[1],
            )

        return execute_manual_download(
            step=input_step,
            windows_downloads_dir=config.windows_downloads_dir,
            output_path_without_extension=output_path_without_extension,
        )

    if isinstance(input_step, Path):
        return ExecutedStep(
            output_path=input_step.path,
            extension=os.path.splitext(input_step.path)[1],
        )

    if isinstance(input_step, Episode):
        raise ValueError("Episode input is not implemented yet.")

    raise ValueError(f"Unsupported input type: {type(input_step).__name__}")


def process_pipelines(loaded_config: LoadedConfig) -> None:
    config = loaded_config.config
    db_path = config.database_file_path
    logger = StepEventLogger(db_path=db_path)
    job_start_timestamp = utc_now()
    n_pipelines = len(config.job.pipelines)

    try:
        for pipeline_index, (pipeline_name, pipeline) in enumerate(
            config.job.pipelines.items()
        ):
            pipeline_start_timestamp = utc_now()
            pipeline_dir = get_pipeline_artifact_dir(config, pipeline_name)
            os.makedirs(pipeline_dir, exist_ok=True)

            previous_step = resolve_initial_step(
                config=config,
                pipeline_name=pipeline_name,
                input_step=pipeline.input,
            )

            total_n_steps = len(pipeline.steps)
            pipeline_metadata_json = json.dumps(pipeline.metadata, sort_keys=True)
            pipeline_input_json = json.dumps(
                pipeline.input.model_dump(mode="json"),
                sort_keys=True,
            )

            for step_index, step in enumerate(pipeline.steps):
                is_last_step = step_index == (total_n_steps - 1)
                is_last_pipeline = pipeline_index == (n_pipelines - 1)
                step_type = step.type
                step_json = json.dumps(step.model_dump(mode="json"), sort_keys=True)
                output_extension = previous_step.extension
                filename = (
                    f"{pipeline_name}_step_{step_index}_{step.__class__.__name__}"
                    f"{output_extension}"
                )

                if is_last_step:
                    os.makedirs(config.output_dir, exist_ok=True)
                    intended_output = os.path.join(
                        config.output_dir,
                        f"{pipeline_name}{output_extension}",
                    )
                else:
                    intended_output = os.path.join(pipeline_dir, filename)

                start_ts = utc_now()
                start_file_info = probe_media_file_info(
                    intended_output, include_hash=False
                )
                logger.log_step_event(
                    event_timestamp=start_ts,
                    chart_path=loaded_config.chart_path,
                    chart_json=loaded_config.chart_json,
                    rendered_config_json=loaded_config.rendered_config_json,
                    job_name=config.job.name,
                    pipeline_index=pipeline_index,
                    n_pipelines=n_pipelines,
                    job_end_timestamp=None,
                    pipeline_name=pipeline_name,
                    pipeline_metadata_json=pipeline_metadata_json,
                    pipeline_input_json=pipeline_input_json,
                    total_n_steps=total_n_steps,
                    pipeline_start_timestamp=pipeline_start_timestamp,
                    pipeline_end_timestamp=None,
                    step_index=step_index,
                    step_type=step_type,
                    step_json=step_json,
                    step_event="start",
                    job_start_timestamp=job_start_timestamp,
                    file_info=start_file_info,
                    error_message=None,
                )

                if not config.full_refresh and os.path.exists(intended_output):
                    existing_file_info = probe_media_file_info(
                        intended_output,
                        include_hash=True,
                    )
                    existing_output_hash = existing_file_info.sha256
                    if existing_output_hash is None:
                        existing_output_hash = hash_file_sha256(intended_output)
                    should_skip = logger.should_skip_step(
                        job_name=config.job.name,
                        pipeline_name=pipeline_name,
                        step_index=step_index,
                        step_json=step_json,
                        output_file_hash=existing_output_hash,
                    )
                    if should_skip:
                        skip_ts = utc_now()
                        pipeline_end_timestamp = skip_ts if is_last_step else None
                        job_end_timestamp = (
                            skip_ts if is_last_step and is_last_pipeline else None
                        )
                        logger.log_step_event(
                            event_timestamp=skip_ts,
                            chart_path=loaded_config.chart_path,
                            chart_json=loaded_config.chart_json,
                            rendered_config_json=loaded_config.rendered_config_json,
                            job_name=config.job.name,
                            pipeline_index=pipeline_index,
                            n_pipelines=n_pipelines,
                            job_end_timestamp=job_end_timestamp,
                            pipeline_name=pipeline_name,
                            pipeline_metadata_json=pipeline_metadata_json,
                            pipeline_input_json=pipeline_input_json,
                            total_n_steps=total_n_steps,
                            pipeline_start_timestamp=pipeline_start_timestamp,
                            pipeline_end_timestamp=pipeline_end_timestamp,
                            step_index=step_index,
                            step_type=step_type,
                            step_json=step_json,
                            step_event="skipped",
                            job_start_timestamp=job_start_timestamp,
                            file_info=existing_file_info,
                            error_message=None,
                        )
                        previous_step = ExecutedStep(
                            output_path=intended_output,
                            extension=output_extension,
                        )
                        continue

                try:
                    if isinstance(step, Trim):
                        previous_step = execute_trim(
                            step=step,
                            previous_step=previous_step,
                            output_path=intended_output,
                        )
                    elif isinstance(step, Interpolate):
                        previous_step = execute_interpolate(
                            step=step,
                            previous_step=previous_step,
                            output_path=intended_output,
                        )
                    elif isinstance(step, Upscale):
                        previous_step = execute_upscale(
                            step=step,
                            previous_step=previous_step,
                            output_path=intended_output,
                        )
                    elif isinstance(step, CopyTracks):
                        previous_step = execute_copy_tracks(
                            step=step,
                            previous_step=previous_step,
                            output_path=intended_output,
                        )
                    else:
                        raise ValueError(f"Unsupported step type: {step_type}")
                except Exception as exc:
                    failed_ts = utc_now()
                    failed_file_info = probe_media_file_info(
                        intended_output, include_hash=False
                    )
                    logger.log_step_event(
                        event_timestamp=failed_ts,
                        chart_path=loaded_config.chart_path,
                        chart_json=loaded_config.chart_json,
                        rendered_config_json=loaded_config.rendered_config_json,
                        job_name=config.job.name,
                        pipeline_index=pipeline_index,
                        n_pipelines=n_pipelines,
                        job_end_timestamp=failed_ts,
                        pipeline_name=pipeline_name,
                        pipeline_metadata_json=pipeline_metadata_json,
                        pipeline_input_json=pipeline_input_json,
                        total_n_steps=total_n_steps,
                        pipeline_start_timestamp=pipeline_start_timestamp,
                        pipeline_end_timestamp=failed_ts,
                        step_index=step_index,
                        step_type=step_type,
                        step_json=step_json,
                        step_event="failed",
                        job_start_timestamp=job_start_timestamp,
                        file_info=failed_file_info,
                        error_message=str(exc),
                    )
                    raise

                completed_ts = utc_now()
                completed_file_info = probe_media_file_info(
                    previous_step.output_path,
                    include_hash=True,
                )
                pipeline_end_timestamp = completed_ts if is_last_step else None
                job_end_timestamp = (
                    completed_ts if is_last_step and is_last_pipeline else None
                )
                logger.log_step_event(
                    event_timestamp=completed_ts,
                    chart_path=loaded_config.chart_path,
                    chart_json=loaded_config.chart_json,
                    rendered_config_json=loaded_config.rendered_config_json,
                    job_name=config.job.name,
                    pipeline_index=pipeline_index,
                    n_pipelines=n_pipelines,
                    job_end_timestamp=job_end_timestamp,
                    pipeline_name=pipeline_name,
                    pipeline_metadata_json=pipeline_metadata_json,
                    pipeline_input_json=pipeline_input_json,
                    total_n_steps=total_n_steps,
                    pipeline_start_timestamp=pipeline_start_timestamp,
                    pipeline_end_timestamp=pipeline_end_timestamp,
                    step_index=step_index,
                    step_type=step_type,
                    step_json=step_json,
                    step_event="completed",
                    job_start_timestamp=job_start_timestamp,
                    file_info=completed_file_info,
                    error_message=None,
                )
    finally:
        logger.close()


def load_config() -> LoadedConfig:
    chart_path = os.environ["CHART_PATH"]
    with open(chart_path, "r") as f:
        chart = Chart.model_validate(yaml.safe_load(f))

    with open(chart.config_template_path, "r") as f:
        rendered_config = jinja2.Template(f.read()).render(**chart.values)

    config = Config.model_validate(yaml.safe_load(rendered_config))
    return LoadedConfig(
        config=config,
        chart_path=chart_path,
        chart_json=json.dumps(chart.model_dump(mode="json"), sort_keys=True),
        rendered_config_json=json.dumps(config.model_dump(mode="json"), sort_keys=True),
    )


def main() -> None:
    loaded_config = load_config()
    process_pipelines(loaded_config)
