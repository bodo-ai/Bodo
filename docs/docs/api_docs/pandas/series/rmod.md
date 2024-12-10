# `pd.Series.rmod`

`pandas.Series.rmod(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument | datatypes |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| `other` | <ul><li> numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar |

!!! note
`Series.rmod` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.rmod(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      0
1      1
2      2
3      1
4      1
      ..
995    5
996    4
997    3
998    2
999    1
Length: 1000, dtype: int64
```
