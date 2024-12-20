# Benchmarks

## NYC Taxi Monthly Trips with Precipitation

For this benchmark, we adapt [an example data science workload](https://github.com/toddwschneider/nyc-taxi-data/blob/c65ad8332a44f49770644b11576c0529b40bbc76/citibike_comparison/analysis/analysis_queries.sql#L1) into a pandas workload that reads from a public S3 bucket and calculates the average trip duration and number of trips based on features like weather conditions, pickup and dropoff location, month, and whether the trip was on a weekday.

### Dataset

The New York City Taxi and Limousine Commission's [For Hire Vehicle High Volume dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (FHVHV) consists of over one billion trips taken by "for hire vehicles" including Uber and Lyft. To get the weather on a given day, we use a separate dataset of [Central Park weather observations](https://github.com/toddwschneider/nyc-taxi-data/blob/c65ad8332a44f49770644b11576c0529b40bbc76/data/central_park_weather.csv). The For Hire Vehicle High Volume dataset consists of 1,036,465,968 rows and 24 columns. The Central Park Weather dataset consists of 5,538 rows and 9 columns.

### Setting

For this benchmark, we use the full FHVHV dataset stored in Parquet files on S3. The total size of this dataset is 24.7 GiB. The Central Park Weather data ia stored in a single CSV file on S3 and its total size is 514 KiB.

We compared Bodo's performance on this workload to other systems including [Dask](https://www.dask.org/), [Modin on Ray](https://docs.ray.io/en/latest/ray-more-libs/modin/index.html), and [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) and observed a speedup of 20-240x. The implementations for all of these systems can be found in [`nyc_taxi`](./nyc_taxi/). Versions of the packages used are summarized below.

| Package      | Version      |
|----------------|----------------|
| bodo   | 2024.10   |
| dask   | 2024.9.1  |
| dask-cloudprovider  | 2024.9.1  |
| modin   | 0.32.0   |
| ray   | 2.40.0   |
| spark  | 3.5.2 |

For cluster creation and configuration, we use the [Bodo SDK](https://docs.bodo.ai/2024.12/guides/using_bodo_platform/bodo_platform_sdk_guide/) for Bodo, Dask Cloud Provider for Dask, Ray for Modin, and AWS EMR for Spark. Scripts to configure and launch clusters for each system can be found in the same directory as the implementation.

Each benchmark is collected on a cluster containing 4 worker instances and 128 physical cores. Dask, Modin, and Spark use 4 `r6i.16xlarge` instances, each consisting of 32 physical cores and 256 GiB of memory. Dask Cloud Provider also allocates an additional `c6i.xlarge` instance for the distributed scheduler which contains 2 cores. Bodo is run on 4 `c6i.16xlarge` instances, each consisting of 32 physical cores and 64 GiB of memory.

### Results

The graph below summarizes the total execution time of each system (averaged over 3 runs). Results were last collected on December 12th, 2024.

<img src="./img/nyc-taxi-benchmark.png" alt="Monthly High Volume for Hire Vehicle Trips with Precipitation Benchmark Execution Time" title="Monthly High Volume for Hire Vehicle Trips with Precipitation Average Execution Time" width="30%">

## Local Benchmark

You can start to see the benefits of using Bodo from your laptop by running the notebooks found in [`./nyc_taxi/notebooks`](./nyc_taxi/notebooks) which include a smaller version of the NYC Taxi Monthly Trips with Precipitation benchmark. To set up, install the required packages using pip in a clean environment that includes Python 3.12:

``` shell
pip install bodo==2024.12.1 "dask[dataframe]"==2024.12.0 "modin[all]"==0.32.0 pyspark==3.5.3 pandas==2.2.3 boto3
```

We use a smaller subset of the [For Hire Vehicle High Volume dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) to allow the workload to run locally (example setup: an Apple M2 Macbook Pro with 10 cores and 16 GB memory). We have also included an implementation using Pandas, which is often on par with or better than other distributed frameworks when data size is smaller due to the overhead from coordinating parallel workers. Even at this smaller scale, Bodo shows a roughly 3x improvement over the next best system (Dask/Pandas).

The results below were collected December 18th, 2024. Note that these numbers might differ based on your specific hardware and operating system.


| System      | Total Execution Time (s)     |
|----------------|----------------|
| Bodo   | 1.007   |
| Dask   | 3.091  |
| Pandas | 3.58 |
| Modin/Ray | 13.65 |
| PySpark   | 27.27   |

To see an even bigger difference, try increasing the number of rows read by specifying a different parquet file such as `s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/fhvhv_tripdata_2019-02.parquet`. On this size (~20 million rows), Spark runs out of memory while Pandas and Dask become 4-5x slower than Bodo.

The code to run the larger dataset is also included in the notebooks section. Results are summarized in the table below.

| System      | Total Execution Time (s)     |
|----------------|----------------|
| Bodo   | 4.228   |
| Pandas | 17.990 |
| Dask   | 21.41  |
| Modin/Ray | 118.52 |
| PySpark   | OOM  |