# `pd.Series.is_monotonic`

`pandas.Series.is_monotonic

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.is_monotonic
>>> S = pd.Series(np.arange(100))
>>> f(S)
True
```

