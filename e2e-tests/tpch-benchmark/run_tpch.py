"""Runs the TPCH queries in our codebase via CREATE TABLE ... AS ... SELECT.

This is used for benchmarking the streaming I/O with snowflake. 

Options are explained by invoking python ./run_tpch.py from any directory. There should be a file named SNOWFLAKE_CREDS in this directory
with the standard bodo snowflake credentials.

Output file is a csv with columns:

database, schema - the database and schema the tpch query was run on
streaming(on/off) - whether the query was run with streaming enabled or not
volcano(on/off) - whether the volcano planner was explicitly requested (note: if streaming is on, then volcano is on)
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
from typing import NamedTuple, Optional

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
    volcano: bool
    local: bool = False
    batch_size: Optional[int] = None

    @property
    def name(self) -> str:
        volcanostr = "volcano" if self.volcano else "novolcano"
        streamingstr = f"streaming-{self.batch_size}" if self.streaming else "batch"
        return f"{self.database}-{self.schema}-{volcanostr}-{streamingstr}"

    @staticmethod
    def _make_csv_header():
        return ["database", "schema", "streaming", "volcano", "batch size"]

    def _make_csv_row(self):
        return [
            self.database,
            self.schema,
            "on" if self.streaming else "off",
            "on" if self.volcano else "off",
            self.batch_size or "N/A",
        ]


def DB_CONFIGS(dbname: str, schema: str) -> list[TestConfig]:
    return [
        TestConfig(database=dbname, schema=schema, streaming=False, volcano=False),
        TestConfig(database=dbname, schema=schema, streaming=True, volcano=True),
        TestConfig(database=dbname, schema=schema, streaming=False, volcano=True),
    ]


CONFIGS = DB_CONFIGS("TEST_DB", "TPCH_SF100")  # + DB_CONFIGS("TEST_DB", "TPCH_SF1000")

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
    "--no-volcano",
    help="Exclude any test configs with volcano on",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--memory-profile",
    help="If given, then a dstat mem profile is taken while each query is run",
    default=False,
    action="store_true",
)

args = parser.parse_args()
print(args.include, args.exclude)

NUM_ITERATIONS = 2


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


with open(args.output, "wt") as output:
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
                                tmp.write(
                                    f"CREATE OR REPLACE TABLE TEST_DB.PUBLIC.test_benchmark_tpch_{query_id} AS ("
                                )
                            with open(qname, "rt") as sql:
                                tmp.write(sql.read())
                            if not test.local:
                                tmp.write(")")
                            tmp.flush()

                            extra_args = []
                            if test.volcano:
                                extra_args.append("--volcano")
                            else:
                                extra_args.append("--no-volcano")
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
                            # Without this, bodo will re-use plans from streaming/non-streaming or volcano/no-volcano
                            env["BODO_PLATFORM_CACHE_LOCATION"] = d
                            print("Using cache: ", d)
                            executable = sys.exeucutable

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
                            except KeyboardInterrupt:
                                raise
                            except:
                                writer.writerow(
                                    test._make_csv_row()
                                    + [i, os.path.basename(qname), "error"]
                                )
