# `pd.Series.ffill`

`pandas.Series.ffill(axis=None, inplace=False, limit=None, downcast=None)`

### Supported Arguments None

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.ffill()
>>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
>>> f(S)
0    <NA>
1       1
2       1
3      -2
4      -2
5       5
6       5
dtype: Int64
```

