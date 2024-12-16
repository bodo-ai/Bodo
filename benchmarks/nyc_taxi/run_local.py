"""Run local version of Bodo, Dask, Modin, and Pyspark benchmarks
Accepts a list of files to use as the High Volume For Hire Vehicle dataset
(defaults to first 5 million rows of the dataset). For exhaustive list of
files, see s3://bodo-example-data/nyc-taxi/fhvhv. You can also optionally
specify the system to run e.g. just run bodo or "all" to run on all systems.

usage:
    python run_local.py --dataset FILE --system SYSTEM
"""

import argparse

from .bodo.nyc_taxi_precipitation import (
    get_monthly_travels_weather as bodo_get_monthly_travels_weather,
)
from .dask.nyc_taxi_precipitation import (
    local_get_monthly_travels_weather as dask_get_monthly_travels_weather,
)
from .modin_ray.nyc_taxi_precipitation import (
    local_get_monthly_travels_weather as modin_get_monthly_travels_weather,
)
from .spark.spark_nyc_taxi_precipitation import (
    get_monthly_travels_weather as spark_get_monthly_travels_weather,
)

hvfhv_small = "s3://bodo-example-data/nyc-taxi/fhvhv_5M_rows.pq"


def get_spark_paths(files: str | list[str]) -> str:
    """Gets the equivalent path for spark to use."""
    if isinstance(files, str):
        return files.replace("s3://", "s3a://").replace("fhvhv/", "fhvhv-rewrite/")

    assert len(files) != 1, "Spark benchmark expects a single path argument."

    return files[0].replace("s3://", "s3a://").replace("fhvhv/", "fhvhv-rewrite/")


def main(dataset: str, system: str):
    if system == "dask" or system == "all":
        print("Running Dask...")
        dask_get_monthly_travels_weather(dataset)

    if system == "modin" or system == "all":
        print("Running Modin...")
        modin_get_monthly_travels_weather(dataset)

    if system == "spark" or system == "all":
        print("Running Spark...")
        spark_get_monthly_travels_weather(get_spark_paths(dataset))

    if system == "bodo" or system == "all":
        print("Running Bodo...")
        bodo_get_monthly_travels_weather(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        nargs="*",
        required=False,
        default=hvfhv_small,
        help="Path to parquet file(s) to use for local benchmark.",
    )
    parser.add_argument(
        "--system",
        "-s",
        required=False,
        default="all",
        help="System to run benchmark on, 'all' runs all systems.",
    )
    args = parser.parse_args()
    dataset = args.dataset
    system = args.system

    main(dataset, system)
