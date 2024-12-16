"""Runs the TPCH queries in our codebase via CREATE TABLE ... AS ... SELECT.

This is used for benchmarking the streaming I/O with snowflake.

Options are explained by invoking python ./run_tpch.py from any directory. There should be a file named SNOWFLAKE_CREDS in this directory
with the standard bodo snowflake credentials.

Output file is a csv with columns:

database, schema - the database and schema the tpch query was run on
streaming(on/off) - whether the query was run with streaming enabled or not
batch size - batch size for streaming IO, or NA if the default or not streaming
N - The iteration number
Query - The name of the query
time - time in seconds
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import NamedTuple

import pandas as pd

import bodo.tests.utils

LOCAL_TABLE_BASE = os.environ.get("TPCH_LOCAL_TABLE_BASE", "")
LOCAL_TABLES = {
    "customer": os.path.join(LOCAL_TABLE_BASE, "customer"),
    "lineitem": os.path.join(LOCAL_TABLE_BASE, "lineitem"),
    "nation": os.path.join(LOCAL_TABLE_BASE, "nation"),
    "orders": os.path.join(LOCAL_TABLE_BASE, "orders"),
    "part": os.path.join(LOCAL_TABLE_BASE, "part"),
    "partsupp": os.path.join(LOCAL_TABLE_BASE, "partsupp"),
    "region": os.path.join(LOCAL_TABLE_BASE, "region"),
    "supplier": os.path.join(LOCAL_TABLE_BASE, "supplier"),
}


class TestConfig(NamedTuple):
    database: str
    schema: str
    streaming: bool
    local: bool = False
    batch_size: int | None = None

    @property
    def name(self) -> str:
        streamingstr = f"streaming-{self.batch_size}" if self.streaming else "batch"
        return f"{self.database}-{self.schema}-{streamingstr}"

    @staticmethod
    def _make_csv_header():
        return ["database", "schema", "streaming", "batch size"]

    def _make_csv_row(self):
        return [
            self.database,
            self.schema,
            "on" if self.streaming else "off",
            self.batch_size or "N/A",
        ]


def DB_CONFIGS(dbname: str, schema: str) -> list[TestConfig]:
    return [
        TestConfig(database=dbname, schema=schema, streaming=False),
        TestConfig(database=dbname, schema=schema, streaming=True),
        TestConfig(database=dbname, schema=schema, streaming=False),
    ]


pattern = os.path.join(
    os.path.dirname(__file__),
    "../../BodoSQL/calcite_sql/bodosql-calcite-application/src/test/resources/com/bodosql/calcite/application/tpch_q*.sql",
)
pattern = os.path.abspath(pattern)
print(pattern)


bodo_sql_wrapper = os.path.join(
    os.path.dirname(__file__),
    "../../buildscripts/build-bodo-from-source-platform/BodoSQLWrapper.py",
)
bodo_sql_wrapper = os.path.abspath(bodo_sql_wrapper)

TIME_RE = re.compile("It took ([0-9]+(.[0-9]*)) seconds")


def get_time(output: str):
    print(output)
    match = TIME_RE.search(output)
    if match is None:
        raise RuntimeError("Could not find string in: \n" + output)
    return float(match[1])


parser = argparse.ArgumentParser(
    prog="run_tpch",
    usage="run_tpch.py [OPTIONS...]",
    description="Runs all TPCH tests against a particular bodo/snowflake configuration and reports the results",
)

parser.add_argument("--output", "-o", required=False, default="./benchmark.csv")
parser.add_argument(
    "--include",
    help="Include tpch tests matching the specified pattern",
    nargs="*",
    default=[],
)
parser.add_argument(
    "--exclude",
    help="Exclude TPCH tests matching the specified pattern",
    nargs="*",
    default=[],
)
parser.add_argument(
    "--ranks",
    help="Number of ranks to run with",
    type=int,
    default="1",
)
parser.add_argument(
    "--no-streaming",
    help="Exclude any test configs with streaming on",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--memory-profile",
    help="If given, then a dstat mem profile is taken while each query is run",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--check-output",
    help="Check the output of each query against the non-streaming baseline and don't write back to snowflake",
    default=False,
    action="store_true",
)

parser.add_argument(
    "--pq-out-dir",
    help="Directory to write parquet output to, only used if --check-output is given",
    default=None,
)

parser.add_argument(
    "--test-schema",
    help="Test schemas to run, supply multiple times to run multiple schemas",
    nargs="*",
    default=["TPCH_SF1"],
    type=str,
    action="extend",
)
parser.add_argument(
    "--test-db", help="Test databases to read from", default="TEST_DB", type=str
)

args = parser.parse_args()
print(args.include, args.exclude)

NUM_ITERATIONS = 2
CONFIGS = []
for schema in args.test_schema:
    CONFIGS.extend(DB_CONFIGS(args.test_db, schema))


@contextmanager
def dstat(args, qname: str):
    if args.memory_profile:
        print("MEMORY PROFILE")
        with subprocess.Popen(
            ["dstat", "-tcmndrs", "--output", f"{qname}-metrics.csv"],
            executable="dstat",
        ) as dstat:
            yield
            dstat.terminate()
    else:
        yield


with open(args.output, "w") as output:
    writer = csv.writer(output)
    writer.writerow(TestConfig._make_csv_header() + ["N", "Query", "Time"])
    for qname in glob.glob(pattern):
        if not os.path.basename(qname).startswith("tpch_q"):
            continue
        if len(args.include) > 0 and all(e not in qname for e in args.include):
            continue
        if len(args.exclude) > 0 and any(e in qname for e in args.exclude):
            continue
        for test in CONFIGS:
            with tempfile.TemporaryDirectory() as d:
                if args.no_streaming and test.streaming:
                    continue
                for i in range(NUM_ITERATIONS):
                    with dstat(args, f"{os.path.basename(qname)}-{test.name}-{i}"):
                        print("Running config", test, qname)
                        with tempfile.NamedTemporaryFile(mode="wt") as tmp:
                            print("WOULD RUN", qname)
                            query_id = os.path.splitext(os.path.basename(qname))[0]
                            if not test.local:
                                if args.check_output:
                                    raise RuntimeError(
                                        "Cannot check output if not writing locally"
                                    )
                                tmp.write(
                                    f"CREATE OR REPLACE TABLE TEST_DB.PUBLIC.test_benchmark_tpch_{query_id} AS ("
                                )
                            with open(qname) as sql:
                                tmp.write(sql.read())
                            if not test.local:
                                tmp.write(")")
                            tmp.flush()

                            extra_args = []
                            if test.streaming:
                                extra_args.append("--streaming")
                                if test.batch_size is not None:
                                    extra_args.extend(
                                        ["--streaming", str(test.batch_size)]
                                    )
                            if test.local:
                                extra_args.append("--local-catalog")
                                for tbl, path in LOCAL_TABLES.items():
                                    extra_args.extend(
                                        ["--table", f"{tbl}:{path}:parquet"]
                                    )
                            if args.check_output:
                                pq_out_dir = (
                                    args.pq_out_dir
                                    if args.pq_out_dir is not None
                                    else TemporaryDirectory()
                                )
                                pq_out_dir_name = (
                                    pq_out_dir
                                    if isinstance(pq_out_dir, str)
                                    else pq_out_dir.name
                                )
                                extra_args.extend(
                                    [
                                        "--pq_out_filename",
                                        os.path.join(
                                            pq_out_dir_name,
                                            f"{test.name}_{query_id}.pq",
                                        ),
                                    ]
                                )
                            full_args = [
                                "python",
                                bodo_sql_wrapper,
                                "-c",
                                "SNOWFLAKE_CREDS",
                                "-f",
                                tmp.name,
                                "-d",
                                test.database,
                                "--sf-schema",
                                test.schema,
                                "-g",
                                f"plan-{test.name}",
                            ] + extra_args
                            print("RUN", full_args)
                            env = dict(**os.environ)
                            # Without this, bodo will re-use plans from streaming/non-streaming
                            env["BODO_PLATFORM_CACHE_LOCATION"] = d
                            print("Using cache: ", d)
                            executable = sys.executable

                            if args.ranks != 1:
                                executable = "mpiexec"
                                full_args = [
                                    "-n",
                                    str(args.ranks),
                                    "-prepend-rank",
                                    sys.executable,
                                ] + full_args[1:]

                            try:
                                with subprocess.Popen(
                                    full_args,
                                    executable=executable,
                                    stdout=subprocess.PIPE,
                                    stdin=subprocess.DEVNULL,
                                    env=env,
                                ) as proc:
                                    stdout, stderr = proc.communicate()
                                    proc.wait()
                                    if proc.returncode != 0:
                                        time = f"error ({proc.returncode})"
                                    else:
                                        time = get_time((stdout).decode("utf-8"))
                                        print("Took ", time, "seconds")
                                    writer.writerow(
                                        test._make_csv_row()
                                        + [i, os.path.basename(qname), time]
                                    )
                                    output.flush()
                            except Exception:
                                writer.writerow(
                                    test._make_csv_row()
                                    + [i, os.path.basename(qname), "error"]
                                )
            if args.check_output:
                outputs = []
                for output in glob.glob(
                    os.path.join(pq_out_dir_name, f"*_{query_id}.pq")
                ):
                    outputs.append(pd.read_parquet(output))
                for i in range(len(outputs) - 1):
                    bodo.tests.utils._test_equal(
                        outputs[i],
                        outputs[i + 1],
                        check_dtype=False,
                        reset_index=True,
                        sort_output=True,
                    )
                if args.pq_out_dir is None:
                    pq_out_dir.cleanup()
