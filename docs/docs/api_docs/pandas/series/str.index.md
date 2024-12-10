# `pd.Series.str.index`

`pandas.Series.str.index(sub, start=0, end=None)`

### Supported Arguments

| argument                    | datatypes                            |
|-----------------------------|--------------------------------------|
| `sub`                       |    String                            |
| `start`                     |    Integer                           |
| `end`                       |    Integer                           |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.index("a", start=1)
>>> S = pd.Series(["Aa3", "cea3", "14a3", " a3", "^ Ea3f", "aaa"])
>>> f(S)
0     1
1     2
2     2
3     1
4     3
5     1
dtype: Int64
```

