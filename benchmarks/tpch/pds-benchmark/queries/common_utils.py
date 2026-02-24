from __future__ import annotations

import re
import sys
from importlib.metadata import version
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING, Any

from linetimer import CodeTimer
from settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import polars as pl

settings = Settings()


def get_table_path(table_name: str) -> str:
    """Return the path to the given table."""
    ext = settings.run.io_type if settings.run.include_io else "parquet"
    # Bodo Change: Parquet path
    multi_part_tables = {"lineitem", "orders", "customer", "part", "partsupp"}
    if settings.scale_factor >= 1000:
        multi_part_tables.add("supplier")
    glob_str = "/*" if ext == "parquet" and table_name in multi_part_tables else ""
    ext = "pq" if ext == "parquet" else ext
    return f"{settings.dataset_base_dir}/{table_name}.{ext}{glob_str}"


def log_query_timing(
    solution: str, version: str, query_number: int, time: float
) -> None:
    settings.paths.timings.mkdir(parents=True, exist_ok=True)

    with (settings.paths.timings / settings.paths.timings_filename).open("a") as f:
        if f.tell() == 0:
            f.write("solution,version,query_number,duration[s],io_type,scale_factor\n")

        line = (
            ",".join(
                [
                    solution,
                    version,
                    str(query_number),
                    str(time),
                    settings.run.io_type,
                    str(settings.scale_factor),
                ]
            )
            + "\n"
        )
        f.write(line)


def on_second_call(func: Any) -> Any:
    def helper(*args: Any, **kwargs: Any) -> Any:
        helper.calls += 1  # type: ignore[attr-defined]
        # first call is outside the function
        # this call must set the result
        if helper.calls == 1:  # type: ignore[attr-defined]
            # include IO will compute the result on the 2nd call
            if not settings.run.include_io:
                helper.result = func(*args, **kwargs)  # type: ignore[attr-defined]
            return helper.result  # type: ignore[attr-defined]

        # second call is in the query, now we set the result
        if settings.run.include_io and helper.calls == 2:  # type: ignore[attr-defined]
            helper.result = func(*args, **kwargs)  # type: ignore[attr-defined]

        return helper.result  # type: ignore[attr-defined]

    helper.calls = 0  # type: ignore[attr-defined]
    helper.result = None  # type: ignore[attr-defined]

    return helper


def execute_all(library_name: str) -> None:
    print(settings.model_dump_json())

    query_numbers = _get_query_numbers(library_name)

    with CodeTimer(name=f"Overall execution of ALL {library_name} queries", unit="s"):
        for i in query_numbers:
            run([sys.executable, "-m", f"queries.{library_name}.q{i}"])


def _get_query_numbers(library_name: str) -> list[int]:
    """Get the query numbers that are implemented for the given library."""
    query_numbers = []

    path = Path(__file__).parent / library_name
    expr = re.compile(r"q(\d+).py$")

    for file in path.iterdir():
        match = expr.search(str(file))
        if match is not None:
            query_numbers.append(int(match.group(1)))

    return sorted(query_numbers)


def run_query_generic(
    query: Callable[..., Any],
    query_number: int,
    library_name: str,
    library_version: str | None = None,
    query_checker: Callable[..., None] | None = None,
) -> None:
    """Execute a query."""
    # execute the query once to ensure we are getting a hot run
    query()
    for _ in range(settings.run.iterations):
        with CodeTimer(
            name=f"Run {library_name} query {query_number}", unit="s"
        ) as timer:
            result = query()

        if settings.run.log_timings:
            log_query_timing(
                solution=library_name,
                version=library_version or version(library_name),
                query_number=query_number,
                time=timer.took,
            )

        if settings.run.check_results:
            if query_checker is None:
                msg = "cannot check results if no query checking function is provided"
                raise ValueError(msg)
            if settings.scale_factor != 1:
                msg = f"cannot check results when scale factor is not 1, got {settings.scale_factor}"
                raise RuntimeError(msg)
            query_checker(result, query_number)

        if settings.run.show_results:
            print(result)


def check_query_result_pl(result: pl.DataFrame, query_number: int) -> None:
    """Assert that the Polars result of the query is correct."""
    from polars.testing import assert_frame_equal

    expected = _get_query_answer_pl(query_number)
    assert_frame_equal(result, expected, check_dtypes=False)


def check_query_result_pd(result: pd.DataFrame, query_number: int) -> None:
    """Assert that the pandas result of the query is correct."""
    from pandas.testing import assert_frame_equal

    expected = _get_query_answer_pd(query_number)
    assert_frame_equal(result.reset_index(drop=True), expected, check_dtype=False)


def _get_query_answer_pl(query: int) -> pl.DataFrame:
    """Read the true answer to the query from disk as a Polars DataFrame."""
    from polars import read_parquet

    path = settings.paths.answers / f"q{query}.parquet"
    return read_parquet(path)


def _get_query_answer_pd(query: int) -> pd.DataFrame:
    """Read the true answer to the query from disk as a pandas DataFrame."""
    from pandas import read_parquet

    path = settings.paths.answers / f"q{query}.parquet"
    return read_parquet(path, dtype_backend="pyarrow")
