# Benchmarks

## Monthly High Volume for Hire Vehicle Trips with Precipitation

For this benchmark, we adapt a [SQL query](https://github.com/toddwschneider/nyc-taxi-data/blob/c65ad8332a44f49770644b11576c0529b40bbc76/citibike_comparison/analysis/analysis_queries.sql#L1) into a pandas workload that reads from a public S3 bucket and calculates the average trip duration and number of trips based on features like weather conditions, pickup and dropoff location, month, and whether the trip was on a weekday.

### Dataset

The New York City Taxi and Limousine Commission's [For Hire Vehicle High Volume dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)(FHVHV) consists of over one billion trips taken by "for hire vehicles" including Uber and Lyft. To get the weather on a given day, we use a separate dataset of [Central Park weather observations](https://github.com/toddwschneider/nyc-taxi-data/blob/c65ad8332a44f49770644b11576c0529b40bbc76/data/central_park_weather.csv). The For Hire Vehicle High Volume dataset consists of 1,036,465,968 rows and 24 columns. The Central Park Weather dataset consists of 5,538 rows and 9 columns. 

### Setting

For this benchmark, we used the full FHVHV dataset stored in parquet files on S3. The total size of this dataset was 24.7 GiB. The Central Park Weather data was stored in a single csv file on S3 and it's total size was 514 KiB.

We compared Bodo's performance on this workload to other systems including [Dask](https://www.dask.org/), [Modin on Ray](https://docs.ray.io/en/latest/ray-more-libs/modin/index.html), and [Pyspark](https://spark.apache.org/docs/latest/api/python/index.html) and observed a speedup of 20-240x. The implementations for all of these systems can be found in [`nyc_taxi`](./nyc_taxi/). Versions of the packages used are summarized below. 

| Package      | Version      |
|----------------|----------------|
| bodo   | 2024.10   |
| dask   | 2024.9.1  |
| dask-cloudprovider  | 2024.9.1  |
| modin   | 0.32.0   |
| ray   | 2.40.0   |
| spark  | 3.5.2 |

For cluster creation and configuration, we used the [Bodo SDK](https://docs.bodo.ai/2024.12/guides/using_bodo_platform/bodo_platform_sdk_guide/) for Bodo, Dask Cloudprovider for Dask, Ray for Modin, and AWS EMR for Spark. Scripts to configure and launch clusters for each system can be found in the same directory as the implementation.

Each benchmark was collected on a cluster containing 4 worker instances and 128 physical cores. Dask, Modin, and Spark used 4 `r6i.16xlarge` instances, each consisting of 32 physical cores and 256 GiB of memory. Dask Cloudprovider also allocated an additional `r6i.16xlarge` instance for the scheduler. Bodo was run on 4 `c6i.16xlarge` instances, each consisting of 32 physical cores and 64 GiB of memory.

### Results

The graph below summarizes the total execution time of each system (averaged over 3 runs). Results were last collected on December 12th, 2024.

<img src="./img/nyc_taxi.png" alt="Monthly High Volume for Hire Vehicle Trips with Precipitation Benchmark Execution Time" title="Monthly High Volume for Hire Vehicle Trips with Precipitation Average Execution Time" width="30%">

## Local Benchmark

You can start to see the benefits of using Bodo from your laptop by running a smaller version of our benchmark locally. To set up, install the required packages using pip:

``` shell
pip install bodo
pip install "dask[dataframe]"
pip install "modin[all]"
pip install pyspark
```
Then run the benchmarks!

``` shell
# (from the benchmarks/ directory)
python -m nyc_taxi.run_local
```

We used a smaller subset of the [For Hire Vehicle High Volume dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) to allow the workload to run locally on an Apple M2 Macbook Pro with 10 cores and 16 GB memory. Even at this smaller scale, Bodo shows a modest improvement over the next best system (Dask). 

To see an even bigger difference, try increasing the number of rows read by specifying a different parquet file such as `s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/fhvhv_tripdata_2019-02.parquet`. On this size (~20 million rows), Dask runs out of memory whereas Bodo continue to run:

``` shell
# run Dask on first parquet file (~20 million rows)
python -m nyc_taxi.run_local -b dask -d s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/fhvhv_tripdata_2019-02.parquet

# run Bodo on first parquet file (~20 million rows)
python -m nyc_taxi.run_local -b bodo -d s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/fhvhv_tripdata_2019-02.parquet
```