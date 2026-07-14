from pathlib import Path
from typing import List

import typer
import yaml
from pipe.ops import (
    build_all_operation_images,
    build_operation_images_by_keys,
    kill_all_operation_containers,
    list_buildable_docker_targets,
)
from pipe.pipeline import (
    filter_config_pipelines,
    load_config_from_chart_path,
    run_config,
)
from pydantic import BaseModel

SETTINGS_FILE = "cli-settings.yaml"
app = typer.Typer()


class CliSettings(BaseModel):
    charts_directory: str


def load_or_create_settings() -> CliSettings:
    settings_path = Path.cwd() / SETTINGS_FILE
    if settings_path.exists():
        with open(settings_path, "r") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            try:
                return CliSettings.model_validate(loaded)
            except Exception:
                pass

    settings_path.touch()
    charts_directory = typer.prompt("Charts directory", default="charts/")
    settings = CliSettings(charts_directory=charts_directory)
    with open(settings_path, "w") as f:
        yaml.safe_dump(settings.model_dump(mode="json"), f, sort_keys=False)
    return settings


def choose_chart(charts_directory: str) -> Path:
    charts_path = Path(charts_directory)
    if not charts_path.exists() or not charts_path.is_dir():
        raise ValueError(
            f"Charts directory does not exist or is not a directory: {charts_directory}"
        )

    chart_files = sorted(
        [
            p
            for p in charts_path.iterdir()
            if p.is_file() and p.suffix.lower() in [".yaml", ".yml"]
        ]
    )
    if not chart_files:
        raise ValueError(f"No chart files found in: {charts_directory}")

    typer.echo("Available charts:")
    for index, chart_file in enumerate(chart_files, start=1):
        typer.echo(f"{index}. {chart_file.name}")

    selected_index = typer.prompt("Select chart number", type=int)
    if selected_index < 1 or selected_index > len(chart_files):
        raise ValueError(f"Invalid chart selection: {selected_index}")

    return chart_files[selected_index - 1]


def _parse_pipeline_selection(selection: str, pipeline_names: List[str]) -> List[str]:
    items = [item.strip() for item in selection.split(",") if item.strip()]
    if not items:
        raise ValueError("No pipeline selection provided.")

    selected: List[str] = []
    for item in items:
        if item.isdigit():
            index = int(item)
            if index < 1 or index > len(pipeline_names):
                raise ValueError(f"Invalid pipeline index: {index}")
            selected.append(pipeline_names[index - 1])
            continue

        if item not in pipeline_names:
            raise ValueError(f"Unknown pipeline name: {item}")
        selected.append(item)

    seen = set()
    deduped: List[str] = []
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def choose_pipelines(pipeline_names: List[str]) -> List[str]:
    typer.echo("Available pipelines:")
    for index, name in enumerate(pipeline_names, start=1):
        typer.echo(f"{index}. {name}")

    selection = typer.prompt(
        "Run all pipelines? [Y] or enter number/name list (comma-separated)",
        default="y",
    ).strip()
    normalized = selection.lower()

    if normalized in ["", "y", "yes", "all", "a", "*"]:
        return pipeline_names

    if normalized in ["n", "no"]:
        selection = typer.prompt(
            "Choose one or more pipelines by number or name (comma-separated)"
        )
    return _parse_pipeline_selection(selection, pipeline_names)


@app.command()
def run(
    chart: str | None = typer.Option(
        None,
        "--chart",
        help="Path to a chart file. If omitted, you will be prompted to select one.",
    ),
) -> None:
    settings = load_or_create_settings()
    if chart is None:
        chart_path = choose_chart(settings.charts_directory)
    else:
        chart_path = Path(chart)
        if not chart_path.exists() or not chart_path.is_file():
            raise ValueError(f"Chart file does not exist: {chart}")

    loaded_config = load_config_from_chart_path(str(chart_path))

    pipeline_names = list(loaded_config.config.job.pipelines.keys())
    selected = choose_pipelines(pipeline_names)

    filtered_config = filter_config_pipelines(loaded_config.config, selected)
    run_config(
        filtered_config,
        chart_path=loaded_config.chart_path,
        chart_json=loaded_config.chart_json,
    )


def _parse_numbered_selection(selection: str, n_items: int) -> List[int]:
    items = [item.strip() for item in selection.split(",") if item.strip()]
    if not items:
        raise ValueError("No selection provided.")

    selected: List[int] = []
    for item in items:
        if not item.isdigit():
            raise ValueError(f"Invalid selection '{item}'. Use numbers like 1,2,3.")
        idx = int(item)
        if idx < 1 or idx > n_items:
            raise ValueError(f"Invalid selection number: {idx}")
        selected.append(idx - 1)

    deduped: List[int] = []
    seen = set()
    for idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    return deduped


@app.command()
def build() -> None:
    targets = list_buildable_docker_targets()
    if not targets:
        typer.echo("No buildable docker targets found.")
        return

    typer.echo("Buildable docker images:")
    for index, (_, image, dockerfile) in enumerate(targets, start=1):
        typer.echo(f"{index}. {image} ({dockerfile})")

    selection = (
        typer.prompt(
            "Build all images? [Y] or enter numbers (comma-separated)",
            default="y",
        )
        .strip()
        .lower()
    )

    if selection in ["", "y", "yes", "all", "a", "*"]:
        build_all_operation_images()
        typer.echo("Built all operation docker images.")
        return

    selected_indexes = _parse_numbered_selection(selection, len(targets))
    selected_keys = [targets[idx][0] for idx in selected_indexes]
    build_operation_images_by_keys(selected_keys)
    typer.echo("Built selected operation docker images.")


@app.command()
def kill() -> None:
    kill_all_operation_containers()
    typer.echo("Kill invoked for all operation docker containers.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
