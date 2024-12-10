# `pd.Series.str.join`

`pandas.Series.str.join(sep)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `sep`                       |    String                              |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.join(",")
>>> S = pd.Series([["a", "fe", "@23"], ["a", "b"], [], ["c"]])
>>> f(S)
0    a,fe,@23
1         a,b
2
3           c
dtype: object
```

