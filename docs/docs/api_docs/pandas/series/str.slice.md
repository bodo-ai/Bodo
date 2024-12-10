# `pd.Series.str.slice`

`pandas.Series.str.slice(start=None, stop=None, step=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|-----------------------------------|
| `start` | Integer |
| `stop` | Integer |
| `step` | Integer |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.slice(1, 4)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0    A
1    c
2    1
3
4    @
5    a
6    #
dtype: object
```
