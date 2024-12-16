# `pd.Series.str.isdigit`

`pandas.Series.str.isdigit()`

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.isdigit()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0    False
1    False
2     True
3    False
4    False
5    False
6    False
dtype: boolean
```

