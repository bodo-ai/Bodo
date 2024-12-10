# `pd.Series.str.startswith`

`pandas.Series.str.startswith(pat, na=None)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `pat`                       |    String                              |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.startswith("A")
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0     True
1    False
2    False
3    False
4    False
5    False
6    False
dtype: boolean
```

