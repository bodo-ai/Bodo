import time

import modin.pandas as pd
import ray
from modin.pandas.io import to_ray


def get_monthly_travels_weather(weather_dataset, hvfhv_dataset):
    start_read = time.time()
    central_park_weather_observations = pd.read_csv(
        weather_dataset, parse_dates=["DATE"], storage_options={"anon": True}
    )
    central_park_weather_observations = central_park_weather_observations.rename(
        columns={"DATE": "date", "PRCP": "precipitation"},
        copy=False,
    )
    fhvhv_tripdata = pd.read_parquet(hvfhv_dataset, storage_options={"anon": True})
    end = time.time()
    print("Reading Time: ", (end - start_read))

    start_compute = time.time()

    central_park_weather_observations["date"] = central_park_weather_observations[
        "date"
    ].dt.date
    fhvhv_tripdata["date"] = fhvhv_tripdata["pickup_datetime"].dt.date
    fhvhv_tripdata["month"] = fhvhv_tripdata["pickup_datetime"].dt.month
    fhvhv_tripdata["hour"] = fhvhv_tripdata["pickup_datetime"].dt.hour
    fhvhv_tripdata["weekday"] = fhvhv_tripdata["pickup_datetime"].dt.dayofweek.isin(
        [0, 1, 2, 3, 4]
    )

    monthly_trips_weather = fhvhv_tripdata.merge(
        central_park_weather_observations, on="date", how="inner"
    )
    monthly_trips_weather["date_with_precipitation"] = (
        monthly_trips_weather["precipitation"] > 0.1
    )

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

    monthly_trips_weather["time_bucket"] = monthly_trips_weather.hour.map(
        get_time_bucket
    )
    monthly_trips_weather = monthly_trips_weather.groupby(
        [
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ],
        as_index=False,
    ).agg({"hvfhs_license_num": "count", "trip_miles": "mean"})
    monthly_trips_weather = monthly_trips_weather.sort_values(
        by=[
            "PULocationID",
            "DOLocationID",
            "month",
            "weekday",
            "date_with_precipitation",
            "time_bucket",
        ]
    )
    monthly_trips_weather = monthly_trips_weather.rename(
        columns={
            "hvfhs_license_num": "trips",
            "trip_miles": "avg_distance",
        },
        copy=False,
    )
    end = time.time()
    print("Monthly Taxi Travel Times Computation Time: ", end - start_compute)

    start_write = time.time()
    monthly_trips_weather_ray = to_ray(monthly_trips_weather)
    monthly_trips_weather_ray.write_parquet("local:///tmp/data/modin_result.pq")
    end = time.time()
    print("Writing time:", (end - start_write))
    print("Total E2E time:", (end - start_read))
    return monthly_trips_weather


if __name__ == "__main__":
    ray.init(address="auto")
    cpu_count = ray.cluster_resources()["CPU"]
    print("RAY CPU COUNT: ", cpu_count)

    weather_dataset = "s3://bodo-example-data/nyc-taxi/central_park_weather.csv"
    hvfhv_dataset = "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/"
    get_monthly_travels_weather(weather_dataset, hvfhv_dataset)
