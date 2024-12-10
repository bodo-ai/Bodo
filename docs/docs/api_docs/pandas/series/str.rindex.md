# `pd.Series.str.rindex`

`pandas.Series.str.index(sub, start=0, end=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|--------------------------------------|
| `sub` | String |
| `start` | Integer |
| `end` | Integer |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.rindex("i")
>>> S = pd.Series(["alphabet soup is delicious", "eieio", "iguana"])
>>> f(S)
0     22
1     3
2     0
dtype: Int64
```
