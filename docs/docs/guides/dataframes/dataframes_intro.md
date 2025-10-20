
<!-- 
To make changes to this page, first make your changes in the notebook: examples/_Tutorials/dataframes_intro.ipynb
then run the script from the project directory: python docs/scripts/notebook_to_docs.py examples/_Tutorials/dataframes_intro.ipynb --outdir docs/docs/guides/dataframes --tag df_devguide
-->


Bodo DataFrames Developer Guide {#df_devguide}
=================

[View Notebook on GitHub](https://github.com/bodo-ai/Bodo/blob/main/examples/_Tutorials/dataframes_intro.ipynb)


This guide provides an introduction to using Bodo DataFrames, explains some of the important concepts and gives tips on how to integrate Bodo DataFrames into existing Pandas workflows.

## Creating BodoDataFrames

You can create a BodoDataFrame by reading data from a file or table using an I/O function. Currently supported IO functions include: 

* `pd.read_parquet`
* `pd.read_iceberg`
* `pd.read_csv`

`pd.read_parquet` and `pd.read_iceberg` are lazy APIs, meaning that no actual data is read until needed in a subsequent operation.

You can also create BodoDataFrames from a Pandas DataFrame using the `from_pandas` function, which is useful when working with third party libraries that return Pandas DataFrames.

### Unsupported DataFrames

Unlike Pandas, BodoDataFrames cannot support arbitrary Python types in columns. Each column in a BodoDataFrame should have a single supported type. Supported types include ints, floats, bool, decimal128, timestamps/datetime, dates, durations/timedelta, string, binary, list, map, and struct. 

Some examples of unsupported DataFrames:

``` py
# Mixed-type tuples are not supported (use structs instead)
DataFrame({"A": [("a", 1), ("b", 2)]})

# Each column must hold a single type
DataFrame({"A": [1, "a"]})

# DataFrames cannot have arbitrary Python objects
DataFrame({"A": [MyObject()] * 4})
```


## Lazy Evaluation and Plans

In Bodo, operations on DataFrames and Series are lazy, meaning that they return a lazy result representing a DataFrame, Series or Scalar which contains some metadata i.e. a schema, but not the actual data itself. Instead, lazy results contain a "plan" attribute. A plan is an expression tree describing how to go from data sources to the current object using relational operators like join, aggregate, or project.

Lazy evaluation allows Bodo to optimize the plan before execution, which can have a huge impact (e.g. 100x) on workload performance. Common optimizations include reordering joins, pushing filters to I/O, or eliminating dead columns.

To see an example of lazy evaluation, we will create a DataFrame, representing a Parquet read over a billion row dataset (NYC taxi). Normally, this dataset would be too large to fit into memory on most laptops, however since the `read_parquet` API is lazy, no actual data is materialized. 


```python
import pandas
import bodo.pandas as pd
from bodo.ext import plan_optimizer

df = pd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_tripdata")
```

We can immediately inspect the metadata of our lazy DataFrame such as the column names and data types. When `read_parquet` is called, Bodo opens the first couple parquet files in the dataset to infer the schema, which is typically pretty fast. 

We can also look at the plan for this DataFrame, which is just a single Parquet scan node.

Finally, we can get the length of the dataset, which executes a small plan to scan the entire dataset and get the row count in each file without pulling any of the rows into memory.


```python
print(df.columns)
print(df.dtypes)

print(df._plan.generate_duckdb().toString())

print(len(df))
```

    Index(['hvfhs_license_num', 'dispatching_base_num', 'originating_base_num',
           'request_datetime', 'on_scene_datetime', 'pickup_datetime',
           'dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_miles',
           'trip_time', 'base_passenger_fare', 'tolls', 'bcf', 'sales_tax',
           'congestion_surcharge', 'airport_fee', 'tips', 'driver_pay',
           'shared_request_flag', 'shared_match_flag', 'access_a_ride_flag',
           'wav_request_flag', 'wav_match_flag'],
          dtype='object')
    hvfhs_license_num              string[pyarrow]
    dispatching_base_num           string[pyarrow]
    originating_base_num           string[pyarrow]
    request_datetime        timestamp[us][pyarrow]
    on_scene_datetime       timestamp[us][pyarrow]
    pickup_datetime         timestamp[us][pyarrow]
    dropoff_datetime        timestamp[us][pyarrow]
    PULocationID                    int64[pyarrow]
    DOLocationID                    int64[pyarrow]
    trip_miles                     double[pyarrow]
    trip_time                       int64[pyarrow]
    base_passenger_fare            double[pyarrow]
    tolls                          double[pyarrow]
    bcf                            double[pyarrow]
    sales_tax                      double[pyarrow]
    congestion_surcharge           double[pyarrow]
    airport_fee                    double[pyarrow]
    tips                           double[pyarrow]
    driver_pay                     double[pyarrow]
    shared_request_flag            string[pyarrow]
    shared_match_flag              string[pyarrow]
    access_a_ride_flag             string[pyarrow]
    wav_request_flag               string[pyarrow]
    wav_match_flag                 string[pyarrow]
    dtype: object
    ┌───────────────────────────┐
    │BODO_READ_PARQUET(HVFH...  │
    │    ────────────────────   │
    │      ~1036465968 Rows     │
    └───────────────────────────┘
    
    1036465968


Bodo can automatically push filters down to IO, which is useful as it avoids materializing extra rows. When we run the optimizer in the example below, two plan nodes (a read into a filter) becomes a single node (a read with a filter).


```python
filt = df[(df['PULocationID'] == 1) & (df['DOLocationID'] == 148)]

print("Before optimizing:")
print(filt._plan.generate_duckdb().toString())

print("After optimizing:")
print(plan_optimizer.py_optimize_plan(filt._plan.generate_duckdb()).toString())
```

    Before optimizing:
    ┌───────────────────────────┐
    │           FILTER          │
    │    ────────────────────   │
    │        Expressions:       │
    │        (#[2.7] = 1)       │
    │       (#[2.8] = 148)      │
    └─────────────┬─────────────┘
    ┌─────────────┴─────────────┐
    │BODO_READ_PARQUET(HVFH...  │
    │    ────────────────────   │
    │      ~1036465968 Rows     │
    └───────────────────────────┘
    
    After optimizing:
    ┌───────────────────────────┐
    │BODO_READ_PARQUET(HVFH...  │
    │    ────────────────────   │
    │          Filters:         │
    │       PULocationID=1      │
    │      DOLocationID=148     │
    │                           │
    │      ~207293193 Rows      │
    └───────────────────────────┘
    


Another optimization that Bodo can do is join reordering. In this example, we want to inner join two dataframes on a key column where one dataframe is much larger than the other. The optimizer swaps the sides of the join to avoid materializing the larger result in memory.


```python
df1 = pd.from_pandas(
    pandas.DataFrame({"A": [1,2,3,4,5]})
)

df2 = pd.from_pandas(pandas.DataFrame(
    {"B": [1,2,3,4,5] * 1000})
)

jn1 = df1.merge(df2, left_on="A", right_on="B")

print("Before optimizing:")
print(jn1._plan.generate_duckdb().toString())

print("After optimizing:")
print(plan_optimizer.py_optimize_plan(jn1._plan.generate_duckdb()).toString())
```

    Before optimizing:
    ┌───────────────────────────┐
    │         PROJECTION        │
    │    ────────────────────   │
    │        Expressions:       │
    │           #[5.0]          │
    │           #[4.0]          │
    └─────────────┬─────────────┘
    ┌─────────────┴─────────────┐
    │      COMPARISON_JOIN      │
    │    ────────────────────   │
    │      Join Type: INNER     │
    │                           ├──────────────┐
    │        Conditions:        │              │
    │     (#[5.0] = #[4.0])     │              │
    └─────────────┬─────────────┘              │
    ┌─────────────┴─────────────┐┌─────────────┴─────────────┐
    │      BODO_READ_DF(A)      ││      BODO_READ_DF(B)      │
    │    ────────────────────   ││    ────────────────────   │
    │          ~5 Rows          ││         ~5000 Rows        │
    └───────────────────────────┘└───────────────────────────┘
    
    After optimizing:
    ┌───────────────────────────┐
    │         PROJECTION        │
    │    ────────────────────   │
    │        Expressions:       │
    │           #[8.0]          │
    │           #[8.0]          │
    │                           │
    │         ~5000 Rows        │
    └─────────────┬─────────────┘
    ┌─────────────┴─────────────┐
    │      COMPARISON_JOIN      │
    │    ────────────────────   │
    │      Join Type: INNER     │
    │                           │
    │        Conditions:        ├──────────────┐
    │     (#[7.0] = #[8.0])     │              │
    │                           │              │
    │         ~5000 Rows        │              │
    └─────────────┬─────────────┘              │
    ┌─────────────┴─────────────┐┌─────────────┴─────────────┐
    │      BODO_READ_DF(B)      ││      BODO_READ_DF(A)      │
    │    ────────────────────   ││    ────────────────────   │
    │         ~5000 Rows        ││          ~5 Rows          │
    └───────────────────────────┘└───────────────────────────┘
    


### Plan execution

Plan optimization typically happens right before execution. The optimized plan is then converted into a sequence of pipelines that are executed in parallel on Bodo workers. Data is streamed through the operators in these pipelines in batches where it is ultimately collected either in a file or an in-memory result. 

Plan execution is triggered by operations like writing to a Parquet file or Iceberg table, printing data, or when an unsupported operation is encountered. The cell below gives examples of operations that return lazy results as well as operations that require collection:


```python
df = pd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_5M_rows.pq")

# Selecting lists of columns returns a lazy DataFrame result
df = df[["PULocationID", "DOLocationID", "base_passenger_fare", "trip_time", "tips", "pickup_datetime"]]

# Selecting a single column returns a lazy BodoSeries
trip_time = df.trip_time

# Reducing a single column to a value produces a lazy BodoScalar
mean_trip_time = trip_time.mean()

# Printing the lazy result triggers execution
print("mean trip time: ", mean_trip_time)

# Writing a lazy dataframe to a file or table triggers execution
df.head(10000).to_parquet("taxi_data.pq")
```

    mean trip time:  1128.3018568


### Fallback for unsupported methods

While Bodo DataFrames supports most common compute intensive operations in Pandas, there are some operations, parameters, or combinations of operations that are not supported yet. [Refer to the DataFrames documentation page](https://docs.bodo.ai/latest/api_docs/dataframe_lib/) for the most up to date list of supported features. Note that while a function may say "supported", there could be a subset of parameters that are not supported yet. 

By default, Bodo automatically raises a warning when an unsupported operation is encountered and falls back to the Pandas implementation, which will typically collect the entire dataset in memory. After the Pandas operation finishes, if the result is a DataFrame or Series, it will be automatically cast back to Bodo so that subsequent operations continue lazily.


```python
df = pd.from_pandas(pandas.DataFrame({"A": range(4), "B": range(1,5)}))

# Unsupported function: transform
df = df.transform(lambda x: x + 1)
print(type(df))

# Subsequent operations will be lazily evaluated
sum_A = df.A.sum()
```

    <class 'bodo.pandas.frame.BodoDataFrame'>


    /Users/scottroutledge/Documents/Bodo/bodo/pandas/utils.py:1215: BodoLibFallbackWarning: transform is not implemented in Bodo DataFrames yet. Falling back to Pandas (may be slow or run out of memory).
      warnings.warn(BodoLibFallbackWarning(msg))


### Using execute_plan to reuse expensive results

Lazy Bodo DataFrames and Series can be explicitly collected using the `execute_plan` function. This function is useful when you have separate plans that depend on the same, expensive computation that produces a small result. In the example below, a larger Taxi dataset is joined with weather observations and aggregated, producing a smaller DataFrame. The result is then referenced in two separate locations.

Without the `execute_plan` call, the plan for `expensive` (read parquet, read csv, join, groupby-aggregate etc) would be executed multiple times for each dependency.


```python
def expensive_computation() -> pd.DataFrame:
    taxi_df = pd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_5M_rows.pq")
    taxi_df["date"] = taxi_df.pickup_datetime.dt.date

    central_park_weather_observations = pd.read_csv(
            "s3://bodo-example-data/nyc-taxi/central_park_weather.csv",
            parse_dates=["DATE"],
        )[["DATE", "PRCP"]]
    central_park_weather_observations["DATE"] = central_park_weather_observations[
        "DATE"
    ].dt.date

    central_park_weather_observations["date_with_precipitation"] = (
            central_park_weather_observations["PRCP"] > 0.1
        )

    taxi_weather = taxi_df.merge(
            central_park_weather_observations,
            left_on="date",
            right_on="DATE",
            how="left",
        )

    weather_tips = taxi_weather.groupby(
            [
                "PULocationID",
                "DOLocationID",
                "date_with_precipitation",
            ],
            as_index=False,
        ).agg(average_tip=("tips", "mean"))

    return weather_tips

expensive = expensive_computation()

expensive.execute_plan()

result1 = print(expensive[expensive["average_tip"] > 1].average_tip.count())

result2 = print(expensive.average_tip.max())
```

    14348
    80.0


## User defined functions

Bodo supports custom transformations on DataFrames and Series using APIs like `Series.map` or `DataFrame.apply`. By default, user defined functions (UDFs) are automatically JIT compiled to avoid Python overheads. In the cell below `get_time_bucket` is compiled eagerly when `map` is called.


```python
df = pd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_5M_rows.pq")

df["hour"] = df.pickup_datetime.dt.hour

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

print(df.hour.map(get_time_bucket).head(5))

```

    0    other
    1    other
    2    other
    3    other
    4    other
    Name: hour, dtype: large_string[pyarrow]


If compilation fails, a warning will be printed and the function will execute in Python mode, which will first run your custom function on a small sample of data to determine output types before continuing lazy evaluation. In the example below, `get_time_bucket` is used as a helper function in a UDF, but the definition is not exposed to JIT, leading to typing errors:


```python
def apply_with_python_fallback(row):
    return get_time_bucket(row.hour)

print(df.apply(apply_with_python_fallback, axis=1).head(5))
```

    /Users/scottroutledge/Documents/Bodo/bodo/pandas/frame.py:1417: BodoCompilationFailedWarning: DataFrame.apply(): Compiling user defined function failed or encountered an unsupported result type. Falling back to Python engine. Add engine='python' to ignore this warning. Original error: [1m[1m[1m[1m[1m[1mDataFrame.apply(): user-defined function not supported: [1mCannot call non-JIT function 'get_time_bucket' from JIT function (convert to JIT or use objmode).[0m
    [1m
    File "../../../../var/folders/w_/z_0_fn150v36jdgzrrlcj8q00000gn/T/ipykernel_57032/4182062319.py", line 2:[0m
    [1m<source missing, REPL/exec in use?>[0m[0m
    [1m
    File "bodo/pandas/frame.py", line 1338:[0m
    [1m        def apply_wrapper_inner(df):
    [1m            return df.apply(func, axis=1, args=args)
    [0m            [1m^[0m[0m
    [0m[0m[0m[1mDuring: Pass bodo_type_inference[0m[0m[0m.
      warnings.warn(BodoCompilationFailedWarning(msg))


    0    other
    1    other
    2    other
    3    other
    4    other
    dtype: string[pyarrow]


If you wish to avoid JIT compilation and run directly in Python mode, you can pass the `engine="python"` argument: 


```python
print(df.apply(apply_with_python_fallback, axis=1, engine='python').head(5))
```

    0    other
    1    other
    2    other
    3    other
    4    other
    dtype: string[pyarrow]


To avoid compilation issues, your function should be type stable, and any helper functions should be decorated with `bodo.jit(spawn=False, distributed=False)`. You can also use most Pandas and Numpy functions inside UDFs.
For additional tips on JIT compilation and troubleshooting, refer to our [Python JIT development guide](https://docs.bodo.ai/latest/quick_start/dev_guide/). 


```python
import bodo

get_time_bucket_jit = bodo.jit(get_time_bucket, distributed=False, spawn=False)

def apply_get_time_bucket(row):
    return get_time_bucket_jit(row.hour)

print(df.apply(apply_get_time_bucket, axis=1).head(5))
```

    0    other
    1    other
    2    other
    3    other
    4    other
    dtype: large_string[pyarrow]


You can also apply custom transformations on groups of data via `groupby.agg` or `groupby.apply`, although currently this features does not support the `engine='python'` argument. If compilation fails, execution will fall back to Pandas.


```python
df = pd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_5M_rows.pq")

def get_small_tip_fraction(tips):
    total_count = len(tips)
    small_tip_count = len(tips[tips < 3])
    return small_tip_count / total_count

agg = df.groupby(['PULocationID', 'DOLocationID']).agg(small_tip_fraction=('tips', get_small_tip_fraction))

agg.head()
```

    /Users/scottroutledge/Documents/Bodo/bodo/pandas/utils.py:1215: BodoLibFallbackWarning: items is not implemented in Bodo DataFrames yet. Falling back to Pandas (may be slow or run out of memory).
      warnings.warn(BodoLibFallbackWarning(msg))





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>small_tip_fraction</th>
    </tr>
    <tr>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>245</th>
      <th>251</th>
      <td>0.956938</td>
    </tr>
    <tr>
      <th>216</th>
      <th>197</th>
      <td>0.986562</td>
    </tr>
    <tr>
      <th>261</th>
      <th>234</th>
      <td>0.876161</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">87</th>
      <th>87</th>
      <td>0.943548</td>
    </tr>
    <tr>
      <th>198</th>
      <td>0.763636</td>
    </tr>
  </tbody>
</table>
</div>



## Migrating Pandas Scripts to Bodo DataFrames

Some general tips for migrating Pandas workflows to Bodo DataFrames:

* **Replace the import one file at a time.**

    Examine individual workflows on a case by case basis to determine if Bodo DataFrames is the right fit.
    If a script uses a lot of unsupported functions or only ever runs on small data sizes, it might be better to keep it in pure Pandas.

* **Measure performance on sufficiently large data sizes.**

    Spawning parallel workers or loading heavy JIT modules can add extra overheads to your program. To get a better feel for the kinds of speedups you can expect when comparing performance vs. Pandas, try running on larger data sizes (e.g. >10,000,000 rows).

* **Run on a sufficiently large machine.**

    To see the benefits of Bodo's parallelism, make sure you are running on a sufficiently large machine or instance with more than one core. 
    Try increasing the number of cores if you need better performance.

