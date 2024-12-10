# `pd.Series.rename`

`pandas.Series.rename(index=None, *, axis=None, copy=True, inplace=False, level=None, errors='ignore')`

### Supported Arguments

| argument | datatypes |
|-------------------------------|-------------------------------------------------------------------------------------|
| `index` | - String |
| `axis` | - Any value. Bodo ignores this argument entirely, which i consistent with Pandas. |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.rename("a")
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      0
1      1
2      2
3      3
4      4
      ..
95    95
96    96
97    97
98    98
99    99
Name: a, Length: 100, dtype: int64
```
