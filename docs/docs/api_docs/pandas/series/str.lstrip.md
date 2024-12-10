# `pd.Series.str.lstrip`

`pandas.Series.str.lstrip(to_strip=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `to_strip` | String |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.lstrip("c")
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0       A
1       e
2      14
3
4       @
5     a n
6    ^ Ef
dtype: object
```
