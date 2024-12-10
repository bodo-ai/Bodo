# `pd.core.groupby.SeriesGroupBy.value_counts`

`pandas.core.groupby.SeriesGroupby.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)`

### Supported Arguments

- `ascending`: boolean
  - **Must be constant at Compile Time**

### Example Usage

```py

>>> @bodo.jit
... def f(S):
...     return S.groupby(level=0).value_counts()
>>> S = pd.Series([1, 2, 24, None] * 5, index = ["421", "f31"] * 10)
>>> f(S)

421  1.0     5
     24.0    5
f31  2.0     5
Name: , dtype: int64
```
