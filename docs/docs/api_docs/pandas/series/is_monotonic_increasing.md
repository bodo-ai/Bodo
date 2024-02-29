# `pd.Series.is_monotonic_increasing`

`pandas.Series.is_monotonic_increasing

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.is_monotonic_increasing
>>> S = pd.Series(np.arange(100))
>>> f(S)
True
```

