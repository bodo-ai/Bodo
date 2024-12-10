# `pd.Series.str.get`

`pandas.Series.str.get(i)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `i`                         |    Integer                             |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.get(1)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0    NaN
1      e
2      4
3    NaN
4    NaN
5
6
dtype: object
```

