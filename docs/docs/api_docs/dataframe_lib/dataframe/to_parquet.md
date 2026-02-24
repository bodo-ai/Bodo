# bodo.pandas.BodoDataFrame.to\_parquet
``` py
BodoDataFrame.to_parquet(path=None, engine="auto", compression="snappy", index=None, partition_cols=None, storage_options=None, row_group_size=-1, **kwargs)
```
**GPU:** ✔ Supported

Write a DataFrame as a Parquet dataset.

<p class="api-header">Parameters</p>

: __path: *str*:__ Output path to write. It can be a local path (e.g. `output.parquet`), AWS S3 (`s3://...`), Azure ALDS (`abfs://...`, `abfss://...`), or GCP GCS (`gcs://...`, `gs://`).

: __compression : *str, default 'snappy'*:__ File compression to use. Can be None, 'snappy', 'gzip', or 'brotli'.

: __row_group_size : *int*:__ Row group size in output Parquet files. -1 allows the backend to choose.

: All other parameters will trigger a fallback to [`pandas.DataFrame.to_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html).


<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

bdf = bd.DataFrame(
    {
        "A": bd.array([1, 2, 3, 7] * 3, "Int64"),
        "B": ["A1", "B1", "C1", "Abc"] * 3,
        "C": bd.array([6, 5, 4] * 4, "Int64"),
    }
)

bdf.to_parquet("output.parquet")
print(bd.read_parquet("output.parquet"))
```

Output:
```
    A    B  C
0   1   A1  6
1   2   B1  5
2   3   C1  4
3   7  Abc  6
4   1   A1  5
5   2   B1  4
6   3   C1  6
7   7  Abc  5
8   1   A1  4
9   2   B1  6
10  3   C1  5
11  7  Abc  4
```

---
