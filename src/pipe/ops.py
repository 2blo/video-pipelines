import yaml
import os
from datetime import timedelta
import subprocess
from pipe.chart import Chart
from pipe.config import Config, Trim
import jinja2


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


with open(os.environ["CHART_PATH"], "r") as f:
    chart = Chart.model_validate(yaml.safe_load(f))

with open(chart.config_template_path, "r") as f:
    config = Config.model_validate(
        yaml.safe_load(jinja2.Template(f.read()).render(**chart.values))
    )


for step in config.steps:
    if isinstance(step, Trim):
        trim(
            input_path=config.input.path,
            output_path=config.output.path,
            start=step.start,
            end=step.end,
        )
