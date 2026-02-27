# DataFrameGroupBy.agg {#frame-agg}
```
DataFrameGroupBy.agg(func=None, engine=None, engine_kwargs=None, **kwargs) -> BodoDataFrame
```
**GPU:** ✔ Supported

!!! tip
    When running on GPU, only the following aggregation functions are accelerated: `sum`, `count`, `mean`, `min`, `max`, `var`, `std`, `size`, `skew`, and `nunique`. Other aggregations, including custom aggregations and UDFs, run on CPU even when a GPU is available.
Apply one or more aggregate functions to groups of data in a BodoDataFrame. This method is the same as `DataFrameGroupBy.aggregate`.

<p class="api-header">Parameters</p>

: __func : *function, str, list, dict or None*:__ Function(s) to use for aggregating the data. Acceptable combinations are:

    * A supported function e.g. `sum`
    * The name of a supported aggregation function e.g. `"sum"`
    * A list of functions, which will be applied to each selected column e.g. `["sum"`, `"count"]`
    * A dictionary mapping column name to aggregate function e.g. `{"col_1": "sum", "col_2": "mean"}`
    * None along with key word arguments specifying Named Aggregates.

    [Refer to our documentation][df-lib-groupby] for aggregate functions that are currently supported in addition to most user defined functions.
    Any other combination of arguments will either fallback to Pandas [`DataFrameGroupBy.agg`](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html#pandas.core.groupby.DataFrameGroupBy.agg) or raise a descriptive error.

: __\*\*kwargs__ Key word arguments are used to create Named Aggregations and should be in the form `new_name=pd.NamedAgg(column_name, function)` or simply `new_name=(column_name, function)`.

!!! note
    The `engine` and `engine_kwargs` parameters are not supported, and will trigger a fallback to Pandas if specified.

<p class="api-header">Returns</p>

: __BodoDataFrame__

<p class="api-header">Examples</p>

``` py
import bodo.pandas as bd

bdf1 = bd.DataFrame({
    "A": ["foo", "foo", "bar", "bar"],
    "C": [1, 2, 3, 4],
    "D": ["A", "A", "C", "D"]
})

bdf2 = bdf1.groupby("A").agg("sum")

print(bdf2)
```
Output:
```
     C   D
A
bar  7  CD
foo  3  AA
```
---
``` py
bdf3 = bdf1.groupby("A").agg(["sum", "count"])

print(bdf3)
```
Output:
```
      C         D
    sum count sum count
A
bar   7     2  CD     2
foo   3     2  AA     2
```
---
``` py
bdf4 = bdf1.groupby("A").agg({"C": "mean", "D": "nunique"})

print(bdf4)
```
Output:
```
       C  D
A
bar  3.5  2
foo  1.5  1
```
---
``` py
bdf5 = bdf1.groupby("A").agg(mean_C=bd.NamedAgg("C", "mean"), sum_D=bd.NamedAgg("D", "sum"))

print(bdf5)
```
Output:
```
     mean_C sum_D
A
bar     3.5    CD
foo     1.5    AA
```

---
