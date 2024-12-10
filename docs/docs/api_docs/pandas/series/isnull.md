# `pd.Series.isnull`

`pandas.Series.isnull()`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.isnull()
>>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
>>> f(S)
0     True
1    False
2     True
3    False
4     True
5    False
6     True
dtype: bool
```

