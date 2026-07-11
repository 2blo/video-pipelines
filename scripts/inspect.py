#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import sys
import tempfile
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
    parser.add_argument(
        "--filter",
        default=os.environ.get("PIPELINE_METRICS_FILTER", ""),
        help="Optional SQL filter expression appended as WHERE <filter>.",
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
    db_path: str | None = None,
    table: str | None = None,
    limit: int | None = None,
    row_filter: str | None = None,
) -> None:
    import duckdb

    if db_path is None or table is None or limit is None or row_filter is None:
        args = parse_args()
        if db_path is None:
            db_path = args.db_path
        if table is None:
            table = args.table
        if limit is None:
            limit = args.limit
        if row_filter is None:
            row_filter = args.filter

    assert db_path is not None
    assert table is not None
    assert limit is not None
    assert row_filter is not None

    table = validate_identifier(table)
    if limit < 1:
        raise ValueError(f"Limit must be >= 1, got: {limit}")

    row_filter = row_filter.strip()
    if ";" in row_filter:
        raise ValueError("Filter must not contain ';'.")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    snapshot_dir: str | None = None
    db_path_in_use = db_path
    opened_from_snapshot = False
    try:
        conn = duckdb.connect(db_path_in_use, read_only=True)
    except Exception as exc:
        message = str(exc)
        lock_conflict = "Could not set lock on file" in message
        if not lock_conflict:
            raise

        snapshot_dir = tempfile.mkdtemp(prefix="duckdb_inspect_")
        snapshot_db_path = os.path.join(snapshot_dir, os.path.basename(db_path))
        shutil.copy2(db_path, snapshot_db_path)

        wal_src = f"{db_path}.wal"
        wal_dst = f"{snapshot_db_path}.wal"
        if os.path.exists(wal_src):
            shutil.copy2(wal_src, wal_dst)

        db_path_in_use = snapshot_db_path
        opened_from_snapshot = True
        conn = duckdb.connect(db_path_in_use)

    if opened_from_snapshot:
        print(
            "Live DB is locked by a writer process; reading from a temporary snapshot: "
            f"{db_path_in_use}"
        )

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
            display(Markdown(f"**filter**: {row_filter or '(none)'}"))
            display(Markdown("### flattened (one level)"))
        except Exception:
            print(f"db_path={db_path}")
            print(f"table={table}")
            print(f"rows={n_rows}")
            print(f"filter={row_filter or '(none)'}")
            print("\n=== flattened (one level) ===")

        where_clause = f"WHERE {row_filter}" if row_filter else ""

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
                CASE
                    WHEN job.start_timestamp IS NOT NULL AND job.end_timestamp IS NOT NULL
                    THEN cast(date_diff('millisecond', job.start_timestamp, job.end_timestamp) / 100 AS bigint) / 10.0
                    ELSE NULL
                END AS job_duration_s,
                pipeline.name AS pipeline_name,
                pipeline.metadata_json AS pipeline_metadata_json,
                pipeline.input_json AS pipeline_input_json,
                pipeline.total_n_steps AS pipeline_total_n_steps,
                pipeline.start_timestamp AS pipeline_start_timestamp,
                pipeline.end_timestamp AS pipeline_end_timestamp,
                CASE
                    WHEN pipeline.start_timestamp IS NOT NULL AND pipeline.end_timestamp IS NOT NULL
                    THEN cast(date_diff('millisecond', pipeline.start_timestamp, pipeline.end_timestamp) / 100 AS bigint) / 10.0
                    ELSE NULL
                END AS pipeline_duration_s,
                step.index AS step_index,
                step.type AS step_type,
                step.event AS step_event,
                step.step_json AS step_step_json,
                --step_start_timestamp,
                --step_end_timestamp,
                --CASE
                --    WHEN step_start_timestamp IS NOT NULL AND step_end_timestamp IS NOT NULL
                --    THEN cast(date_diff('millisecond', step_start_timestamp, step_end_timestamp) / 100 AS bigint) / 10.0
                --    ELSE NULL
                --END AS step_duration_s,
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
            {where_clause}
            ORDER BY event_timestamp desc
            LIMIT {limit}
            """
        )
    finally:
        conn.close()
        if snapshot_dir is not None:
            shutil.rmtree(snapshot_dir, ignore_errors=True)


def main() -> None:
    row_filter = """--sql
    job.end_timestamp IS NOT NULL
    """
    query_tables(limit=6, row_filter=row_filter)


if __name__ == "__main__":
    main()
