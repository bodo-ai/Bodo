Bodo DataFrames Developer Guide {#df_devguide}
=================

[View this page as a notebook on GitHub](https://github.com/bodo-ai/Bodo/blob/main/examples/#Tutorials/dataframes_intro.ipynb)

This guide provides an introduction to using Bodo DataFrames, explains some of the important concepts and gives tips on how to integrate Bodo DataFrames into existing Pandas workflows.

## Creating BodoDataFrames

You can create a BodoDataFrame by reading data from a file or table using an I/O function. Currently supported IO functions include:

* `pd.read_parquet`
* `pd.read_iceberg`
* `pd.read_csv`

`pd.read_parquet` and `pd.read_iceberg` are lazy APIs, meaning that no actual data is read until it is needed in a subsequent operation.

You can also create BodoDataFrames from a Pandas DataFrame using the `from_pandas` function, which is useful when working with third party libraries that return Pandas DataFrames.

### Unsupported DataFrames

Unlike Pandas, BodoDataFrames cannot support arbitrary Python types in columns. Each column in a BodoDataFrame should have a single, well defined type. Supported types include ints, floats, bool, decimal128, timestamps/datetime, dates, durations/timedelta, string, binary, list, map, and struct.

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

In Bodo, operations on DataFrames and Series are lazy, meaning that they return a lazy result representing a DataFrame, Series or Scalar which contains some metadata i.e. a schema, but not the actual data itself. Instead, lazy results contain a "plan" attribute, which is an expression tree describing how to go from the data sources to the final object using relational operators like join, aggregate, or project.

Lazy evaluation allows Bodo to optimize the expression tree before execution, which can have a huge impact (e.g. 100x) on workload performance. Common optimizations include reordering joins, pushing filters to I/O, or eliminating dead columns.

To see an example of lazy evaluation, let's create a DataFrame, representing a Parquet read over a billion row dataset (NYC taxi). Normally, this dataset would be too large to fit into memory on most laptops, however since the `read_parquet` API is lazy, no actual data is materialized at this point.


```python
import pandas
import bodo.pandas as pd
from bodo.ext import plan_optimizer

df = pd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_tripdata")
```


    ---------------------------------------------------------------------------

    BodoError                                 Traceback (most recent call last)

    Cell In[1], line 5
          2 import bodo.pandas as pd
          3 from bodo.ext import plan_optimizer
    ----> 5 df = pd.read_parquet("s3://bodo-example-data/nyc-taxi/fhvhv_tripdata")


    File ~/Documents/Bodo/bodo/pandas/utils.py:430, in check_args_fallback.<locals>.decorator.<locals>.wrapper(*args, **kwargs)
        428 try:
        429     start_time = time.perf_counter()
    --> 430     ret = func(*args, **kwargs)
        431     global top_time
        432     time_this_call = time.perf_counter() - start_time


    File ~/Documents/Bodo/bodo/pandas/base.py:159, in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
        157 # Read Parquet schema
        158 use_hive = True
    --> 159 pq_dataset = get_parquet_dataset(
        160     path,
        161     get_row_counts=False,
        162     storage_options=storage_options,
        163     partitioning="hive" if use_hive else None,
        164 )
        165 pq_dataset = parquet_dataset_unify_nulls(pq_dataset)
        166 arrow_schema = pq_dataset.schema


    File ~/Documents/Bodo/bodo/io/parquet_pio.py:859, in get_parquet_dataset(fpath, get_row_counts, filters, storage_options, read_categories, tot_rows_to_read, typing_pa_schema, partitioning)
        857 if isinstance(dataset_or_err, Exception):  # pragma: no cover
        858     error = dataset_or_err
    --> 859     raise error
        860 dataset = pt.cast(ParquetDataset, dataset_or_err)
        862 # As mentioned above, we don't want to broadcast the filesystem because it
        863 # adds time (so initially we didn't include it in the dataset). We add
        864 # it to the dataset now that it's been broadcasted


    BodoError: error from pyarrow: OSError: When getting information for key 'nyc-taxi/fhvhv_tripdata' in bucket 'bodo-example-data': AWS Error UNKNOWN (HTTP status 400) during HeadObject operation: No response body.



We can immediately inspect the metadata of our lazy DataFrame, such as the column names and data types. When `read_parquet` is called, Bodo opens the first couple parquet files in the dataset to infer the schema, which is typically pretty fast.

We can also look at the plan for this DataFrame. Bodo uses DuckDB plans as an intermediary representation to perform optimizations using the DuckDB optimizer.

Finally, we can get the length of the dataset, which executes a small plan which scans the entire dataset, getting the row count in each file without pulling any of the rows into memory.


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


Using the DuckDB optimizer, Bodo can automatically push filters down to IO, which is useful as it avoids materializing extra rows. Notice how, when we run the optimizer, two plan nodes (a read into a filter) becomes a single node (a read with a filter).


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



Another optimization that Bodo can do is join reordering. In this example, we want to inner join two dataframes on a key column where one dataframe is much larger than the other. The optimizer recognizes that it is better to swap the sides of the join to avoid materializing the larger result in memory.


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

Plan optimization happens right before execution. After the plan is optimized, it gets converted into a sequence of pipelines that are executed using parallel workers. Data is streamed through operators in these pipelines in batches.

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

While Bodo DataFrames supports most common compute intensive operations in Pandas, there are some operations, parameters, or combinations of operations that are not supported yet. [Refer to the DataFrames documentation page]() for the most up to date list of supported features. Note that while a function might say it is supported, there may be a subset of parameters that are not supported yet.

By default, Bodo automatically raises a warning when an unsupported operations are encountered and falls back to the Pandas implementation, which will typically collect the entire dataset in memory. After the Pandas operation finishes, if the result is a DataFrame or Series, it will be automatically cast back to Bodo so that subsequent operations continue to be lazily evaluated.


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


## User defined functions

Bodo supports custom transformations on DataFrames and Series using APIs like `Series.map` or `DataFrame.apply`. By default, user defined functions (UDFs) are automatically JIT compiled to avoid Python overheads. In the cell below `get_time_bucket` is compiled eagerly when `map` is called.


```python
import bodo

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
    0    other
    1    other
    2    other
    3    other
    4    other
    dtype: large_string[pyarrow]


    /Users/scottroutledge/Documents/Bodo/bodo/pandas/frame.py:1417: BodoCompilationFailedWarning: DataFrame.apply(): Compiling user defined function failed or encountered an unsupported result type. Falling back to Python engine. Add engine='python' to ignore this warning. Original error: [1m[1m[1m[1m[1m[1mDataFrame.apply(): user-defined function not supported: [1mCannot call non-JIT function 'get_time_bucket' from JIT function (convert to JIT or use objmode).[0m
    [1m
    File "../../../../var/folders/w_/z_0_fn150v36jdgzrrlcj8q00000gn/T/ipykernel_35430/316969206.py", line 33:[0m
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
    0    other
    1    other
    2    other
    3    other
    4    other
    dtype: string[pyarrow]


If compilation fails, a warning will be printed and the function will execute in Python mode, which will first run your custom function on a small sample of data to determine output types. In the example below, `get_time_bucket` is used as a helper function in a UDF, but the definition is not exposed to JIT, leading to typing errors:


```python
def apply_with_python_fallback(row):
    return get_time_bucket(row.hour)

print(df.apply(apply_with_python_fallback, axis=1).head(5))
```

If you wish to avoid JIT compilation and run directly in Python mode, you can pass the `engine="python"` argument:


```python
print(df.apply(apply_with_python_fallback, axis=1, engine='python').head(5))
```

To avoid compilation issues, your function should be type stable, and any helper functions should be decorated with `bodo.jit(spawn=False, distributed=False)`. You can also use most Pandas and Numpy functions inside UDFs.
For additional tips on JIT compilation and troubleshooting, refer to our [Python JIT development guide](https://docs.bodo.ai/latest/quick_start/dev_guide/).


```python
get_time_bucket_jit = bodo.jit(get_time_bucket, distributed=False, spawn=False)

def apply_get_time_bucket(row):
    return get_time_bucket_jit(row.hour)

print(df.apply(apply_get_time_bucket, axis=1).head(5))
```

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

                               small_tip_fraction
    PULocationID DOLocationID
    245          251                     0.956938
    216          197                     0.986562
    261          234                     0.876161
    87           87                      0.943548
                 198                     0.763636


    /Users/scottroutledge/Documents/Bodo/bodo/pandas/utils.py:1215: BodoLibFallbackWarning: items is not implemented in Bodo DataFrames yet. Falling back to Pandas (may be slow or run out of memory).
      warnings.warn(BodoLibFallbackWarning(msg))


## Migrating Pandas Scripts to Bodo DataFrames

Some general tips for migrating Pandas scripts to Bodo DataFrames:

* **Replace the import one file at a time.**

   Examine individual workflows on a case by case basis to determine if Bodo DataFrames is the right fit.
   If a script uses a lot of unsupported functions or only ever runs on small data sizes, it might be better to keep it in pure Pandas.

* **Measure performance on sufficiently large data sizes.**

  Spawning parallel workers or loading heavy JIT modules can add extra overheads to your program. To get a better feel for the kinds of speedups you can expect when comparing performance vs. Pandas, try running on larger data sizes (e.g. >10,000,000 rows).

* **Run on a sufficiently large machine.**

    To see the benefits of Bodo's parallelism, make sure you are running on a sufficiently large instance with more than one core.
    Try increasing the number of cores if you need better performance.

* **Avoid loading JIT if possible.**

  APIs like `map` and `apply` load JIT modules, which can add extra overheads the first time they are called. Consider if your custom function application can be rewritten using builtin functions.
