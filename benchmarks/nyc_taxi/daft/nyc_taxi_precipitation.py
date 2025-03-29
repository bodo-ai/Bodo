"""
Benchmark Daft on a Ray cluster, writing output to an S3 bucket.

usage:
To run on all 4 nodes, do:
    python nyc_taxi_precipitation.py --s3_bucket BUCKET

To run on a single node and write output to head node, do:
    python nyc_taxi_precipitation.py --single_node
"""

import argparse
import time
from dataclasses import dataclass

import daft
import ray
from daft import col


@dataclass(frozen=True)
class Config:
    weather_dataset: str = "s3://bodo-example-data/nyc-taxi/central_park_weather.csv"
    hvfhv_dataset: str = "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/**"
    output_path: str = "monthly_trips_weather.pq"
    io_config: daft.io.IOConfig | None = None


def get_monthly_travels_weather(
    weather_dataset_path: str,
    hvfhv_dataset_path: str,
    output_path: str,
    io_config: daft.io.IOConfig,
) -> daft.DataFrame:
    start = time.time()

    # read data, rename some columns
    central_park_weather_observations = daft.read_csv(
        weather_dataset_path, io_config=io_config
    )
    central_park_weather_observations = (
        central_park_weather_observations.with_columns_renamed(
            {"DATE": "date", "PRCP": "precipitation"}
        )
    )
    hvfhv_dataset = daft.read_parquet(hvfhv_dataset_path, io_config=io_config)

    # parse dates
    central_park_weather_observations = central_park_weather_observations.with_column(
        "date",
        central_park_weather_observations["date"].dt.date(),
    )
    hvfhv_dataset = hvfhv_dataset.with_columns(
        {
            "date": col("pickup_datetime").dt.date(),
            "month": col("pickup_datetime").dt.month(),
            "hour": col("pickup_datetime").dt.hour(),
            "weekday": col("pickup_datetime").dt.day_of_week().is_in([0, 1, 2, 3, 4]),
        }
    )

    # merge with weather observations
    monthly_trips_weather = hvfhv_dataset.join(
        central_park_weather_observations, on="date", how="inner"
    )
    monthly_trips_weather = monthly_trips_weather.with_column(
        "date_with_precipitation", col("precipitation") > 0.1
    )

    # place rides in bucket determined by hour of the day
    def get_time_bucket(t):
        bucket = "other"
        if t in (8, 9, 10):
            bucket = "morning"
        elif t in (11, 12, 13, 14, 15):
            bucket = "midday"
        elif t in (16, 17, 18):
            bucket = "afternoon"
        elif t in (19, 20, 21):
            bucket = "evening"
        return bucket

    monthly_trips_weather = monthly_trips_weather.with_column(
        "time_bucket",
        col("hour").apply(get_time_bucket, return_dtype=daft.DataType.string()),
    )

    # get total trips and average distance for all trips
    monthly_trips_weather = monthly_trips_weather.groupby(
        [
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ]
    ).agg(col("hvfhs_license_num").count(), col("trip_miles").mean())

    monthly_trips_weather = monthly_trips_weather.sort(
        by=[
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ]
    )

    monthly_trips_weather = monthly_trips_weather.with_columns_renamed(
        {
            "hvfhs_license_num": "trips",
            "trip_miles": "avg_distance",
        },
    )

    # write output to S3
    monthly_trips_weather.write_parquet(output_path, write_mode="overwrite")

    end = time.time()
    print("Total E2E time:", (end - start))

    return monthly_trips_weather


def get_config(args) -> Config:
    assert args.s3_bucket or args.single_node, (
        "must specify S3 bucket in distributed case."
    )

    output_path = (
        "monthly_trips_weather.pq"
        if args.single_node
        else f"s3://{args.s3_bucket}/monthly_trips_weather.pq"
    )
    # run single node in anonymous mode to avoid credentials not found issue with Ray.
    io_config = (
        daft.io.IOConfig(s3=daft.io.S3Config(anonymous=True))
        if args.single_node and not args.s3_bucket
        else None
    )

    return Config(
        output_path=output_path,
        io_config=io_config,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3_bucket",
        type=str,
        required=False,
        help="The name of a valid S3 bucket or prefix to write output to.",
    )
    parser.add_argument(
        "--single_node",
        action="store_true",
        required=False,
        help="Flag indicating whether we are executing benchmark on a single node.",
    )
    args = parser.parse_args()

    config = get_config(args)

    # start ray cluster, configure Daft
    ray.init(address="auto")
    daft.context.set_runner_ray()

    get_monthly_travels_weather(
        config.weather_dataset,
        config.hvfhv_dataset,
        config.output_path,
        config.io_config,
    )


if __name__ == "__main__":
    main()
