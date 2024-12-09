# `pd.Series.str.len`

`pandas.Series.str.len()`

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.len()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0    1
1    2
2    2
3    1
4    1
5    3
6    4
dtype: Int64
```

