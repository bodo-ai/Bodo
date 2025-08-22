# SeriesGroupBy.apply {#series-apply}
```
SeriesGroupBy.apply(self, func, *args, include_groups=True, **kwargs)
```

Apply a function group-wise and combine results together.

The function must take a series as the first argument and return a scalar, dataframe or series.
Currently, Bodo DataFrames will fallback to [pandas.core.groupby.SeriesGroupBy.apply](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.apply.html) if the function returns a dataFrame or series.

<p class="api-header">Parameters</p>

: __func : *callable*__: A callable that takes a series as its first argument.
Currently only scalar return types are supported in Bodo DataFrames,
if *func* returns a series or dataframe, a fallback to pandas.core.groupby.SeriesGroupBy.apply will be triggered.

: __include_groups : *bool, default True*__ Whether to include grouping keys in the input to *func*.
Bodo DataFrames only supports the value False, if *include_groups*, a fallback to pandas.core.groupby.SeriesGroupBy.apply will be triggered.

: __args, kwargs__ Positional and keyword arguments to pass to *func*.
Currently, passing arguments to *func* is not supported and will trigger a fallback to pandas.core.groupby.SeriesGroupBy.apply.

<p class="api-header">Returns</p>

: __BodoSeries__ or __pandas.DataFrame__ depending on the value of *as_index*.

<p class="api-header">Example</p>

``` py
import bodo.pandas as bd

df = bd.DataFrame({'A': 'a a b'.split(),

                   'B': [1, 2, 3],

                   'C': [4, 6, 5]})

apply_res = df.groupby('A')['B'].apply(lambda x: x.min() - x.max())
print(apply_res)
```

Output:
```
A
a    -1
b     0
Name: B, dtype: int64[pyarrow]
```