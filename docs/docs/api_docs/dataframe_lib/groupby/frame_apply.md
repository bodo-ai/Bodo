# DataFrameGroupBy.apply {#frame-apply}
```
DataFrameGroupBy.apply(self, func, *args, include_groups=False, **kwargs)
```

Apply a function group-wise and combine results together.

The function must take a dataframe as the first argument and return a scalar, dataframe or series.
Currently, Bodo DataFrames will fallback to Pandas' [DataFrameGroupBy.apply](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.apply.html) if the function returns a DataFrame or Series.

<p class="api-header">Parameters</p>

: __func : *callable*__: A callable that takes a dataframe as a first arguemnt.
Currently only scalar return types are supported in Bodo DataFrames,
if *func* returns a Series or DataFrame, a fallback to `DataFrameGroupBy.apply` will be triggered.

: __include_groups : *bool, default False*__ Whether to include grouping keys in the input to *func*,
Bodo DataFrames only supports False, if *include_groups*, a fallback to `DataFrameGroupBy.apply` will be triggered.

: __args : tuple__ Positional arguments to pass to *func*.

: __kwargs : dict__ Keyword arguments to pass to *func*.

<p class="api-header">Returns</p>

: __BodoSeries__ or __pandas.DataFrame__ if *func* returns a Series or DataFrame.

<p class="api-header">Examples</p>

``` py
import bodo.pandas as bd

df = bd.DataFrame({'A': 'a a b'.split(),

                   'B': [1, 2, 3],

                   'C': [4, 6, 5]})

apply_res = df.groupby('A')[['B', 'C']].apply(lambda df: df.B.min() - df.C.max())
print(apply_res)
```

Output:
```
A
a   -5
b   -2
dtype: int64
```
