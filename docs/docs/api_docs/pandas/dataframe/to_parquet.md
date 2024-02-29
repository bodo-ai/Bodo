# `pd.DataFrame.to_parquet`

`pandas.DataFrame.to_parquet(path, engine='auto', compression='snappy', index=None, partition_cols=None, storage_options=None)`

### Supported Arguments
* `path` is a required argument and must be a string. When writing distributed dataframes, the path refers to a directory of parquet files.
* `engine` argument only supports `"auto"` and `"pyarrow"`. Default: `"auto"` which uses the pyarrow engine.
* `compression` argument must be one of: `"snappy"`, `"gzip"`, `"brotli"`, `None`. Default: `"snappy"`.
* `index` argument must be a constant bool or `None`. Default: `None`.
* `partition_cols` argument is supported in most cases, except when the columns in the DataFrame cannot be determined at compile time. This must be a list of column names or `None`. Default: `None`.
* `storage_options` argument supports only the default value `None`.
* `row_group_size` argument can be used to specify the maximum size of the row-groups in the generated parquet files; the actual size of the written row-groups may be smaller then this value. This must be an integer. If not specified, Bodo writes row-groups with 1M rows.

!!! note
    Bodo writes multiple files in parallel (one per core), and the total number of row-groups across all files is roughly `max(num_cores, total_rows / row_group_size)`.
    The size of the row groups can affect read performance significantly. In general, the dataset should have at least as many row-groups as the number of cores used for reading, but ideally a lot more.
    At the same time, the row-groups shouldn't be too small since this can lead to overheads at read time.
    For more details, refer to the [parquet file format](https://parquet.apache.org/docs/concepts/){target=blank}.


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
...   df.to_parquet("dataset.pq")
>>> f()
```


