import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import List, Literal

import duckdb
import ffmpeg
import jinja2
import yaml
from pipe.chart import Chart
from pipe.config import (
    Config,
    Episode,
    Ffmpeg,
    Interpolate,
    ManualDownload,
    Path,
    Upscale,
)
from pipe.ops import (
    ExecutedStep,
    execute_ffmpeg,
    execute_interpolate,
    execute_manual_download,
    get_ffmpeg_step_extension,
    execute_upscale,
)
from pydantic import BaseModel


StepEventType = Literal["start", "completed", "skipped", "failed"]
STEP_EVENTS_TABLE = "step_events_v3"


class LoadedConfig(BaseModel):
    config: Config
    chart_path: str
    chart_json: str
    rendered_config_json: str


class MediaFileInfo(BaseModel):
    path: str
    extension: str
    size_bytes: int | None
    width: int | None
    height: int | None
    fps: float | None
    duration_ms: int | None
    frame_count: int | None
    sha256: str | None
    metadata_text: str | None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _parse_fraction_to_float(value: str | None) -> float | None:
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_file_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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

    duration_seconds: float | None = None
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
        job_end_timestamp: datetime | None,
        pipeline_name: str,
        pipeline_metadata_json: str,
        pipeline_input_json: str,
        total_n_steps: int,
        pipeline_start_timestamp: datetime,
        pipeline_end_timestamp: datetime | None,
        step_index: int,
        step_type: str,
        step_json: str,
        step_event: StepEventType,
        job_start_timestamp: datetime,
        file_info: MediaFileInfo,
        error_message: str | None,
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
        if input_step.show not in config.shows:
            raise ValueError(f"Unknown show in episode input: {input_step.show}")

        show = config.shows[input_step.show]
        try:
            relative_path = show.episode_pattern.format(
                input_step.season,
                input_step.episode,
            )
        except Exception:
            relative_path = show.episode_pattern.format(
                season=input_step.season,
                episode=input_step.episode,
            )

        resolved_path = os.path.join(show.path, relative_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                "Episode source file not found: "
                f"{resolved_path} "
                f"(show={input_step.show}, season={input_step.season}, episode={input_step.episode})"
            )

        return ExecutedStep(
            output_path=resolved_path,
            extension=os.path.splitext(resolved_path)[1],
        )

    raise ValueError(f"Unsupported input type: {type(input_step).__name__}")


def run_config(
    config: Config,
    *,
    chart_path: str = "",
    chart_json: str = "{}",
    rendered_config_json: str | None = None,
) -> None:
    logger = StepEventLogger(db_path=config.database_file_path)
    job_start_timestamp = utc_now()
    n_pipelines = len(config.job.pipelines)
    rendered = rendered_config_json
    if rendered is None:
        rendered = json.dumps(config.model_dump(mode="json"), sort_keys=True)

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
                if isinstance(step, Ffmpeg):
                    output_extension = get_ffmpeg_step_extension(
                        step, previous_step.extension
                    )
                else:
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
                    chart_path=chart_path,
                    chart_json=chart_json,
                    rendered_config_json=rendered,
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
                            chart_path=chart_path,
                            chart_json=chart_json,
                            rendered_config_json=rendered,
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
                    if isinstance(step, Ffmpeg):
                        previous_step = execute_ffmpeg(
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
                    else:
                        raise ValueError(f"Unsupported step type: {step_type}")
                except Exception as exc:
                    failed_ts = utc_now()
                    failed_file_info = probe_media_file_info(
                        intended_output, include_hash=False
                    )
                    logger.log_step_event(
                        event_timestamp=failed_ts,
                        chart_path=chart_path,
                        chart_json=chart_json,
                        rendered_config_json=rendered,
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
                    chart_path=chart_path,
                    chart_json=chart_json,
                    rendered_config_json=rendered,
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


def load_config_from_chart_path(chart_path: str) -> LoadedConfig:
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


def load_config_from_env() -> LoadedConfig:
    return load_config_from_chart_path(os.environ["CHART_PATH"])


def filter_config_pipelines(config: Config, selected_pipelines: List[str]) -> Config:
    selected = set(selected_pipelines)
    filtered = {
        pipeline_name: pipeline
        for pipeline_name, pipeline in config.job.pipelines.items()
        if pipeline_name in selected
    }
    updated = config.model_copy(deep=True)
    updated.job.pipelines = filtered
    return updated
