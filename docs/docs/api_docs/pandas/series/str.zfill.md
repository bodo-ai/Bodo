# `pd.Series.str.zfill`

`pandas.Series.str.zfill(width)`

### Supported Arguments

| argument | datatypes |
|----------|-----------|
| `width` | Integer |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.zfill(5)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0    0000A
1    000ce
2    00014
3    0000
4    0000@
5    00a n
6    0^ Ef
dtype: object
```
