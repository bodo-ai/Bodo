# `pd.Series.str.count`

`pandas.Series.str.count(pat, flags=0)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|---------------------------------------|
| `pat` | String |
| `flags` | Integer |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.count("w")
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0    1
1    2
2    3
3    2
4    0
5    2
6    2
dtype: Int64
```
