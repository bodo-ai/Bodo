# `pd.Series.str.find`

`pandas.Series.str.find(sub, start=0, end=None)`

### Supported Arguments

| argument                    | datatypes                            |
|-----------------------------|--------------------------------------|
| `sub`                       |    String                            |
| `start`                     |    Integer                           |
| `end`                       |    Integer                           |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.find("a3", start=1)
>>> S = pd.Series(["Aa3", "cea3", "14a3", " a3", "a3@", "a n3", "^ Ea3f"])
>>> f(S)
0     1
1     2
2     2
3     1
4    -1
5    -1
6     3
dtype: Int64
```

