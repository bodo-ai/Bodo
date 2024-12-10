# `pd.Series.str.extractall`

`pandas.Series.str.extractall(pat, flags=0)`

### Supported Arguments

| argument | datatypes | other requirements |
|----------|-----------|--------------------------------------|
| `pat` | String | **Must be constant at Compile Time** |
| `flags` | Integer | **Must be constant at Compile Time** |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.extractall("(a|n)")
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
         0
  match
0 0      a
3 0      a
5 0      a
  1      n
```
