# `pd.Series.rfloordiv`

`pandas.Series.rfloordiv(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument | datatypes |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| `other` | <ul><li> numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar |

!!! note
`Series.rfloordiv` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.rfloordiv(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      1000
1       499
2       332
3       249
4       199
      ...
995       0
996       0
997       0
998       0
999       0
Length: 1000, dtype: int64
```
