# `pd.Series.dtypes`

`pandas.Series.dtypes`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dtypes
>>> S = pd.Series(np.arange(1000))
>>> f(S)
dtype('int64')
```

