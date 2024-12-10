# `pd.Series.str.isalpha`

`pandas.Series.str.isalpha()`

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.isalpha()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0     True
1     True
2    False
3    False
4    False
5    False
6    False
dtype: boolean
```

