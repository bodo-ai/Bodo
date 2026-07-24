import argparse
import datetime
import functools
import inspect
import os
import time
import warnings
from collections.abc import Callable

import pandas as pd

import bodo.pandas
import bodo.spawn.spawner as spawner
from bodosql import BodoSQLContext, FileSystemCatalog  # noqa


def timethis(
    q: Callable,
    name: str | None = None,
    log_file: str | None = None,
    query: int | None = None,
):
    @functools.wraps(q)
    def wrapped(*args, **kwargs):
        t = time.time()
        result = q(*args, **kwargs)
        msg = name or f"{q.__name__.upper()} Execution time (s):"
        total_time = time.time() - t
        if log_file:
            with open(log_file, "a") as f:
                f.write(
                    f"bodo,{query},{os.environ.get('BODO_NUM_WORKERS', 4)},{total_time:f}\n"
                )
        print(f"{msg} {total_time:f}")
        return result

    return wrapped


_query_to_args: dict[int, list[str]] = {}


def collect_datasets(func: Callable):
    _query_to_args[int(func.__name__[1:])] = list(inspect.signature(func).parameters)
    return func


show_output = False


def exec_func(res):
    global show_output

    if show_output:
        print(res)
    return res


def run_queries(
    root: str,
    queries: list[int],
    scale_factor: float,
    backend,
    warmup: bool,
    n_iters: int = 1,
    log_file: str | None = None,
    answers_path: str | None = None,
    output_path: str | None = None,
    use_stats: bool = False,
):
    if backend is bodo.pandas and bodo.dataframe_library_run_parallel:
        spawner.submit_func_to_workers(lambda: warnings.filterwarnings("ignore"), [])

    total_start = time.time()
    n_passed = 0
    failed_queries = []
    tpch_data = FileSystemCatalog(root)
    for query in queries:
        print(f"Running query {query} at {datetime.datetime.now()}...")
        q = globals()[f"q{query:02}"]

        def query_func():
            return q(tpch_data)

        query_func = timethis(
            query_func,
            name=f"Q{query:02} Execution time (including read_parquet) (s):",
            log_file=log_file,
            query=query,
        )

        try:
            if warmup:
                # Warm up run:
                result = query_func()

            # Second run for timing:
            for _ in range(n_iters):
                result = query_func()

                if answers_path:
                    from bodo.tests.utils import _test_equal

                    answer_df = pd.read_parquet(
                        f"{answers_path}/q{query:02}.pq", dtype_backend="pyarrow"
                    )
                    answer_df = answer_df[
                        list(result.columns)
                    ]  # reorder columns to match result
                    _test_equal(result, answer_df, sort_output=True, reset_index=True)

                if output_path:
                    result.to_parquet(f"{output_path}/q{query:02}.pq")
        except Exception as e:
            print(f"Error running query {query}: {e}")
            failed_queries.append(query)
        else:
            n_passed += 1

    print(f"Total query execution time (s): {time.time() - total_start}")
    print(f"Total successful queries: {n_passed}/{len(queries)}")
    if failed_queries:
        print(f"Failed queries: {failed_queries}")


def create_queries(queries, sql_dir="../sql"):
    for q in queries:
        nn = f"{q:02d}"  # zero-padded two-digit string
        sql_path = os.path.join(sql_dir, f"q{nn}.sql")

        # read SQL file
        with open(sql_path, encoding="utf-8") as f:
            sql_text = f.read()

        func_name = f"tpch_q{nn}"

        # Build the function source string
        func_src = (
            f"""
def {func_name}(tpch_data):
    tpch_query = """
            + "'''\\\n"
            + sql_text
            + "\\\n'''\n"
            + """
    bc = BodoSQLContext(catalog=tpch_data, default_tz=None)
    bodosql_output = bc.sql(tpch_query, None, None, {})
    return bodosql_output
"""
        )

        # Execute into provided globals (or module globals)
        exec(func_src, globals())

        query_func_src = f"""
@timethis
@collect_datasets
def q{nn}(tpch_data):
    return exec_func(tpch_q{nn}(tpch_data))
"""

        exec(query_func_src, globals())


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--folder",
        type=str,
        default="s3://bodo-example-data/tpch/SF1",
        help="The folder containing TPCH data",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="Space separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        required=False,
        default=1.0,
        help="Scale factor (used in query 11).",
    )
    parser.add_argument(
        "--show_output",
        action="store_true",
        required=False,
        help="Whether to print the output.",
    )
    parser.add_argument(
        "--no_warmup",
        action="store_true",
        required=False,
        help="Whether to do warmup run.",
    )
    parser.add_argument(
        "--use_stats",
        action="store_true",
        required=False,
        help="Whether to use json stats files.",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        required=False,
        default=1,
        help="Number of iterations to run each query.",
    )
    parser.add_argument(
        "--log_timings",
        type=str,
        required=False,
        help="File to log timings.",
    )
    parser.add_argument(
        "--answers_path",
        type=str,
        required=False,
        help="Path to diectory containing pre-computed answers (in parquet format), expects names like q<query>.pq",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help="Path to directory to write query outputs (in parquet format), will write files like q<query>.pq",
    )
    args = parser.parse_args()
    data_set = args.folder
    scale_factor = args.scale_factor

    global show_output
    show_output = args.show_output
    do_warmup = not args.no_warmup
    use_stats = args.use_stats

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")
    create_queries(queries)

    warnings.filterwarnings("ignore")

    if args.log_timings is not None:
        if not os.path.exists(args.log_timings):
            with open(args.log_timings, "w") as f:
                f.write("implementation,query,n_gpus,execution_time\n")

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    backend_module = bodo.pandas

    print("Running bodo.pandas: GPU enabled?: ", bodo.gpu_enabled)
    # warmup GPU cluster
    # print(backend_module.DataFrame({"A": [1, 2, 3]})["A"])

    run_queries(
        data_set,
        queries=queries,
        scale_factor=scale_factor,
        backend=backend_module,
        warmup=do_warmup,
        n_iters=args.n_iters,
        log_file=args.log_timings,
        answers_path=args.answers_path,
        output_path=args.output_path,
        use_stats=use_stats,
    )


if __name__ == "__main__":
    print("Running Bodosql TPC-H")
    main()
