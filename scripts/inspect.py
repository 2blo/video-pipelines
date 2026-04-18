#!/usr/bin/env python3

import argparse
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect all records from a DuckDB table."
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("PIPELINE_METRICS_DB_PATH", ".video_pipelines.duckdb"),
        help="Path to DuckDB database file.",
    )
    parser.add_argument(
        "--table",
        default=os.environ.get("PIPELINE_METRICS_TABLE", "step_events_v3"),
        help="Table name to read from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("PIPELINE_METRICS_LIMIT", "100")),
        help="Maximum number of rows to display.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid table name: {value}")
    return value


def should_hide_column(column_name: str) -> bool:
    lowered = column_name.lower()
    hidden_tokens = [
        "dump",
        "json",
        "metadata",
        "raw_media",
        "path",
    ]
    return any(token in lowered for token in hidden_tokens)


def query_tables(
    db_path: str | None = None, table: str | None = None, limit: int | None = None
) -> None:
    import duckdb

    if db_path is None or table is None or limit is None:
        args = parse_args()
        if db_path is None:
            db_path = args.db_path
        if table is None:
            table = args.table
        if limit is None:
            limit = args.limit

    assert db_path is not None
    assert table is not None
    assert limit is not None

    table = validate_identifier(table)
    if limit < 1:
        raise ValueError(f"Limit must be >= 1, got: {limit}")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = duckdb.connect(db_path, read_only=True)
    try:
        count_result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        n_rows = int(count_result[0]) if count_result else 0

        def show_query(query: str) -> None:
            relation = conn.sql(query)
            try:
                import pandas as pd
                from IPython.display import HTML, display

                with pd.option_context(
                    "display.max_columns",
                    None,
                    "display.max_colwidth",
                    None,
                    "display.width",
                    None,
                    "display.max_rows",
                    None,
                ):
                    df = relation.df()
                    visible_columns = [
                        column
                        for column in df.columns
                        if not should_hide_column(str(column))
                    ]
                    if visible_columns:
                        df = df[visible_columns]

                display(
                    HTML(
                        "<div style='max-height:70vh; overflow:auto; border:1px solid #ddd; padding:6px'>"
                        f"{df.to_html(index=False)}"
                        "</div>"
                    )
                )
            except Exception:
                relation.show(max_rows=limit, max_width=0, max_col_width=0)

        try:
            from IPython.display import Markdown, display

            display(Markdown(f"**db_path**: {db_path}"))
            display(Markdown(f"**table**: {table}"))
            display(Markdown(f"**rows**: {n_rows}"))
            display(Markdown("### flattened (one level)"))
        except Exception:
            print(f"db_path={db_path}")
            print(f"table={table}")
            print(f"rows={n_rows}")
            print("\n=== flattened (one level) ===")

        show_query(
            f"""
            SELECT
                event_timestamp,
                raw_media_metadata,
                job.name AS job_name,
                job.chart_path AS job_chart_path,
                job.chart_json AS job_chart_json,
                job.rendered_config_json AS job_rendered_config_json,
                job.pipeline_index AS job_pipeline_index,
                job.n_pipelines AS job_n_pipelines,
                job.start_timestamp AS job_start_timestamp,
                job.end_timestamp AS job_end_timestamp,
                pipeline.name AS pipeline_name,
                pipeline.metadata_json AS pipeline_metadata_json,
                pipeline.input_json AS pipeline_input_json,
                pipeline.total_n_steps AS pipeline_total_n_steps,
                pipeline.start_timestamp AS pipeline_start_timestamp,
                pipeline.end_timestamp AS pipeline_end_timestamp,
                step.index AS step_index,
                step.type AS step_type,
                step.event AS step_event,
                step.step_json AS step_step_json,
                step.error_message AS step_error_message,
                file.path AS file_path,
                file.extension AS file_extension,
                cast(file.size_bytes / 1024 / 1024 AS bigint) AS file_size_mb,
                file.width AS file_width,
                file.height AS file_height,
                file.fps AS file_fps,
                cast(file.duration_ms / 100 AS bigint) / 10.0  AS file_duration_s,
                file.frame_count AS file_frame_count,
                file.sha256 AS file_sha256
            FROM {table}
            ORDER BY event_timestamp desc
            LIMIT {limit}
            """
        )
    finally:
        conn.close()


def main() -> None:
    query_tables(limit=10)


if __name__ == "__main__":
    main()
