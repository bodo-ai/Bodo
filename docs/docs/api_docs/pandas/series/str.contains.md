# `pd.Series.str.contains`

`pandas.Series.str.contains(pat, case=True, flags=0, na=None, regex=True)`

### Supported Arguments

| argument | datatypes | other requirements                   |
|----------|-----------|--------------------------------------|
| `pat`    | String    |                                      |
| `case`   | Boolean   | **Must be constant at Compile Time** |
| `flags`  | Integer   |                                      |
| `regex`  | Boolean   | **Must be constant at Compile Time** |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.contains("a.+")
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0    False
1    False
2    False
3     True
4    False
5     True
6    False
dtype: boolean
```

