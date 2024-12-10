# `pd.Series.str.extract`

`pandas.Series.str.extract(pat, flags=0, expand=True)`

### Supported Arguments

| argument | datatypes | other requirements |
|----------|-------------|--------------------------------------|
| `pat` | - String | **Must be constant at Compile Time** |
| `flags` | - Integer | **Must be constant at Compile Time** |
| `expand` | - Boolean | **Must be constant at Compile Time** |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.extract("(a|e)")
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
     0
0    a
1    e
2  NaN
3    a
4  NaN
5    a
6  NaN
```
