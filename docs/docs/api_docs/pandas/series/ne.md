# `pd.Series.ne`

`pandas.Series.ne(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument | datatypes |
|--------------|-----------------------------------------------------------------------------------------------------------|
| `other` | <ul><li> numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar |

!!! note
`Series.ne` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.ne(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      True
1      True
2      True
3      True
4      True
      ...
995    True
996    True
997    True
998    True
999    True
Length: 1000, dtype: bool
```
