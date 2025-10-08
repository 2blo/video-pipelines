import yaml
import os
from datetime import timedelta
import subprocess
from pipe.chart import Chart
from pipe.config import Config, Trim, ManualDownload, Path
from pydantic import BaseModel
import jinja2
import webbrowser
from time import sleep
import shutil


class ExecutedStep(BaseModel):
    output_path: str
    extension: str


def execute_manual_download(
    step: ManualDownload, windows_downloads_dir: str, output_path_without_extension: str
) -> ExecutedStep:
    files_before = set(os.listdir(windows_downloads_dir))
    webbrowser.open(step.link)
    new_file = None
    while not new_file:
        files_after = set(os.listdir(windows_downloads_dir))
        new_files = files_after - files_before
        if new_files:
            new_file = new_files.pop()
        else:
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


with open(os.environ["CHART_PATH"], "r") as f:
    chart = Chart.model_validate(yaml.safe_load(f))

with open(chart.config_template_path, "r") as f:
    config = Config.model_validate(
        yaml.safe_load(jinja2.Template(f.read()).render(**chart.values))
    )


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
