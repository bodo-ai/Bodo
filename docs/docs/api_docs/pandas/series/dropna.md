# `pd.Series.dropna`

`pandas.Series.dropna(axis=0, inplace=False, how=None)`

### Supported Arguments None

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dropna()
>>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
>>> f(S)
1     1
3    -2
5     5
dtype: Int64
```

