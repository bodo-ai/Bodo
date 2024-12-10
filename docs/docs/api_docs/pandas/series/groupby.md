# `pd.Series.groupby`

`pandas.Series.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)`

### Supported Arguments

| argument | datatypes | other requirements |
|----------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `by` | Array-like or Series data. This is not supported with Decimal or Categorical data. | **Must be constant at Compile Time** |
| `level` | integer | <ul><li> **Must be constant at Compile Time** </li><li> Only `level=0` is supported and not with MultiIndex. |

You must provide exactly one of `by` and `level`

### Example Usage

```py
>>> @bodo.jit
... def f(S, by_series):
...     return S.groupby(by_series).count()
>>> S = pd.Series([1, 2, 24, None] * 5)
>>> by_series = pd.Series(["421", "f31"] * 10)
>>> f(S, by_series)
>
421    10
f31     5
Name: , dtype: int64
```

!!! note
`Series.groupby` doesn't currently keep the name of the original
Series.
