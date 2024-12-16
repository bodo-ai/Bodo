# `pd.Series.notnull`

`pandas.Series.notnull()`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.notnull()
>>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
>>> f(S)
0    False
1     True
2    False
3     True
4    False
5     True
6    False
dtype: bool
```

