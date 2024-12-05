import time

import ray

ray.init(num_cpus=4)
import modin.pandas as pd


def run_modin():
    start = time.time()
    central_park_weather_observations = pd.read_csv("weather.csv", parse_dates=["DATE"])
    central_park_weather_observations = central_park_weather_observations.rename(
        columns={"DATE": "date", "PRCP": "precipitation"}, copy=False
    )
    fhvhv_tripdata = pd.read_parquet("taxi_1k.pq")
    end = time.time()
    print("Reading Time: ", (end - start))

    start = time.time()

    central_park_weather_observations["date"] = central_park_weather_observations[
        "date"
    ].dt.date
    fhvhv_tripdata["date"] = fhvhv_tripdata["pickup_datetime"].dt.date
    fhvhv_tripdata["month"] = fhvhv_tripdata["pickup_datetime"].dt.month
    fhvhv_tripdata["hour"] = fhvhv_tripdata["pickup_datetime"].dt.hour
    fhvhv_tripdata["weekday"] = fhvhv_tripdata["pickup_datetime"].dt.dayofweek.isin(
        [1, 2, 3, 4, 5]
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
    print("Monthly Taxi Travel Times Computation Time: ", end - start)
    return monthly_trips_weather


if __name__ == "__main__":
    result = run_modin()
