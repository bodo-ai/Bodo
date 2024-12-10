# `pd.Series.floordiv`

`pandas.Series.floordiv(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument | datatypes |
|--------------|------------------------------------------------------------------------------------------------------------|
| `other` | <ul><li> numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar |

!!! note
`Series.floordiv` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.floordiv(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0         0
1         0
2         0
3         0
4         0
      ...
995     199
996     249
997     332
998     499
999    1000
Length: 1000, dtype: int64
```
