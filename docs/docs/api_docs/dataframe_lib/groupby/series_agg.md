# SeriesGroupBy.agg {#series-agg}
```
SeriesGroupBy.agg(func=None, engine=None, engine_kwargs=None, **kwargs) -> BodoDataFrame | BodoSeries
```
**GPU:** ✔ Supported

!!! tip
    GPU acceleration for `SeriesGroupBy.agg` is only available for a subset of aggregation functions:
    `sum`, `count`, `mean`, `min`, `max`, `var`, `std`, `size`, `skew`, and `nunique`.
    Other aggregations, including custom aggregations and user-defined functions (UDFs),
    are executed on CPU and may trigger a fallback to the non-GPU engine.
Apply one or more aggregate functions to groups of data in a single column from a BodoDataFrame. This method is the same as `SeriesGroupBy.aggregate`.

<p class="api-header">Parameters</p>

: __func : *function, str, list, dict or None*:__ Function(s) to use for aggregating the data. Acceptable combinations are:

    * A supported function e.g. `sum`
    * The name of a supported aggregation function e.g. `"sum"`
    * A list of functions, which will be applied to each selected column e.g. `["sum"`, `"count"]`
    * None along with key word arguments specifying the supported functions to apply.

    While providing a dictionary argument for *func* is supported, this use has been deprecated in Pandas and will raise an error in newer versions.
    [Refer to our documentation][df-lib-groupby] for aggregate functions that are currently supported in addition to most user defined functions.
    Any other combination of arguments will either fallback to Pandas [`SeriesGroupBy.agg`](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.agg.html#pandas.core.groupby.SeriesGroupBy.agg) or raise a descriptive error.

: __\*\*kwargs__ Key word arguments are used to create Named Aggregations and should be in the form `new_name="function"`.

!!! note
    The `engine` and `engine_kwargs` parameters are not supported, and will trigger a fallback to Pandas if specified.

<p class="api-header">Returns</p>

: __BodoDataFrame__ or __BodoSeries__, depending on the value of *func*.

<p class="api-header">Examples</p>


``` py
import bodo.pandas as bd

bdf1 = bd.DataFrame({
    "A": ["foo", "foo", "bar", "bar"],
    "C": [1, 2, 3, 4],
    "D": ["A", "A", "C", "D"]
})

bdf2 = bdf1.groupby("A")["C"].agg("sum")

print(bdf2)
```
Output:
```
A
bar    7
foo    3
Name: C, dtype: int64[pyarrow]
```
---
``` py
bdf3 = bdf1.groupby("A")["C"].agg(["sum", "count"])

print(bdf3)
```
Output:
```
     sum  count
A
bar    7      2
foo    3      2
```
---
``` py
bdf4 = bdf1.groupby("A")["C"].agg(sum_C="sum", mean_C="mean")

print(bdf4)
```
Output:
```
     sum_C  mean_C
A
bar      7     3.5
foo      3     1.5
```
