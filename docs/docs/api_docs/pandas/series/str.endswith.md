# `pd.Series.str.endswith`

`pandas.Series.str.endswith(pat, na=None)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `pat`                       |    String                              |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.endswith("e")
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0    False
1     True
2    False
3    False
4    False
5    False
6    False
dtype: boolean
```

