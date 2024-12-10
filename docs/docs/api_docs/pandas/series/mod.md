# `pd.Series.mod`

`pandas.Series.mod(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument | datatypes |
|--------------|------------------------------------------------------------------------------------------------------------|
| `other` | <ul><li> numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar |

!!! note
`Series.mod` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.mod(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      1
1      2
2      3
3      4
4      5
      ..
995    1
996    1
997    2
998    1
999    0
Length: 1000, dtype: int64
```
