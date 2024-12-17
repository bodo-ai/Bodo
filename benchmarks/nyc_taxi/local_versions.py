"""Run local version of Bodo, Dask, Modin, and Pyspark benchmarks
Accepts a list of files to use as the High Volume For Hire Vehicle dataset
(defaults to first 5 million rows of the dataset). For exhaustive list of
files, see s3://bodo-example-data/nyc-taxi/fhvhv. You can also optionally
specify the system to run e.g. just run bodo or "all" to run on all systems.

usage:
    python run_local.py --dataset FILE --system SYSTEM
"""

import argparse
import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from nyc_taxi.bodo.nyc_taxi_precipitation import (
    get_monthly_travels_weather as bodo_get_monthly_travels_weather,
)
from nyc_taxi.dask.nyc_taxi_precipitation import (
    local_get_monthly_travels_weather as dask_get_monthly_travels_weather,
)
from nyc_taxi.modin_ray.nyc_taxi_precipitation import (
    local_get_monthly_travels_weather as modin_get_monthly_travels_weather,
)
from nyc_taxi.spark.spark_nyc_taxi_precipitation import (
    get_monthly_travels_weather as spark_get_monthly_travels_weather,
)

SMALL_DATASET_PATH_S3 = "nyc-taxi/fhvhv_5M_rows.pq"
WEATHER_DATASET_PATH_S3 = "nyc-taxi/central_park_weather.csv"
BUCKET_NAME = "bodo-example-data"


def download_data_s3(path_to_s3: str, local_data_dir="data") -> str:
    """Download the dataset from S3 if already exists, skip download."""
    file_name = path_to_s3.split("/", -1)[1]
    local_path = os.path.join(local_data_dir, file_name)

    if os.path.exists(local_path):
        return local_path

    print("Downloading dataset from S3...")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    if not os.path.exists(local_data_dir):
        os.mkdir(local_data_dir)

    s3.download_file(BUCKET_NAME, path_to_s3, local_path)
    return local_path


def main(fhvhv_path_s3: str, system: str):
    weather_path = download_data_s3(WEATHER_DATASET_PATH_S3)
    fhvhv_path = download_data_s3(fhvhv_path_s3)

    get_monthly_travels_weather_impls = {
        "bodo": bodo_get_monthly_travels_weather,
        "dask": dask_get_monthly_travels_weather,
        "modin": modin_get_monthly_travels_weather,
        "spark": spark_get_monthly_travels_weather,
    }

    get_monthly_travels_weather = get_monthly_travels_weather_impls[system]

    print(f"Running {system.capitalize()}...")
    get_monthly_travels_weather(weather_path, fhvhv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        required=False,
        default=SMALL_DATASET_PATH_S3,
        help="Path to parquet file(s) to use for local benchmark.",
    )
    parser.add_argument(
        "--system",
        "-s",
        required=False,
        default="bodo",
        help="System to run benchmark on.",
    )
    args = parser.parse_args()
    fhvhv_path_s3 = args.dataset
    system = args.system

    main(fhvhv_path_s3, system)
