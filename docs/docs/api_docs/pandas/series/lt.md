# `pd.Series.lt`

`pandas.Series.lt(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument | datatypes |
|--------------|-----------------------------------------------------------------------------------------------------------|
| `other` | <ul><li> numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar |

!!! note
`Series.lt` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.lt(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0       True
1       True
2       True
3       True
4       True
      ...
995    False
996    False
997    False
998    False
999    False
Length: 1000, dtype: bool
```
