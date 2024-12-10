# `pd.Series.sub`

`pandas.Series.sub(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|------------------------------------------------------------------------------------------------------------|
| `other` | <ul><li> numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar |

!!! note
`Series.sub` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.sub(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0     -999
1     -997
2     -995
3     -993
4     -991
      ...
995    991
996    993
997    995
998    997
999    999
Length: 1000, dtype: int64
```
