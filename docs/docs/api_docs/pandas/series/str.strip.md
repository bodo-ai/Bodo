# `pd.Series.str.strip`

`pandas.Series.str.strip(to_strip=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `to_strip` | String |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.strip("n")
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0       A
1      ce
2      14
3
4       @
5      a
6    ^ Ef
dtype: object
```
