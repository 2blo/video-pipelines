import yaml
import os
from datetime import timedelta
import subprocess
from pipe.chart import Chart
from pipe.config import Config, Trim, ManualDownload, Path, Upscale
from pydantic import BaseModel
import jinja2
import webbrowser
from time import sleep
import shutil
from typing import Dict


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


def upscale(input_path: str, output_path: str, width: int) -> None:
    command = [
        "python",
        "dependencies/seedvr2/inference_cli.py",
        input_path,
        "--resolution",
        str(width),
        "--output",
        output_path,
    ]
    subprocess.run(command, check=True)


def execute_upscale(
    step: Upscale, previous_step: ExecutedStep, output_path: str
) -> ExecutedStep:
    upscale(
        input_path=previous_step.output_path,
        output_path=output_path,
        width=step.width,
    )
    return ExecutedStep(output_path=output_path, extension=previous_step.extension)


def process_pipelines(config: Config) -> None:
    for pipeline_name, pipeline in config.pipelines.items():
        if isinstance(pipeline.input, ManualDownload):
            previous_step = execute_manual_download(
                step=pipeline.input,
                windows_downloads_dir=config.windows_downloads_dir,
                output_path_without_extension=os.path.join(
                    config.output_dir, pipeline_name, pipeline.input.__class__.__name__
                ),
            )
        elif isinstance(pipeline.input, Path):
            previous_step = ExecutedStep(
                output_path=pipeline.input.path,
                extension=os.path.splitext(pipeline.input.path)[1],
            )
        for i, step in enumerate(pipeline.steps):
            if isinstance(step, Trim):
                previous_step = execute_trim(
                    step=step,
                    previous_step=previous_step,
                    output_path=os.path.join(
                        config.output_dir,
                        pipeline_name,
                        f"step_{i}_{step.__class__.__name__}{previous_step.extension}",
                    ),
                )
            elif isinstance(step, Upscale):
                previous_step = execute_upscale(
                    step=step,
                    previous_step=previous_step,
                    output_path=os.path.join(
                        config.output_dir,
                        pipeline_name,
                        f"step_{i}_{step.__class__.__name__}{previous_step.extension}",
                    ),
                )


def load_config() -> Config:
    with open(os.environ["CHART_PATH"], "r") as f:
        chart = Chart.model_validate(yaml.safe_load(f))

    with open(chart.config_template_path, "r") as f:
        config = Config.model_validate(
            yaml.safe_load(jinja2.Template(f.read()).render(**chart.values))
        )
    return config

def main() -> None:
    config = load_config()
    process_pipelines(config)
