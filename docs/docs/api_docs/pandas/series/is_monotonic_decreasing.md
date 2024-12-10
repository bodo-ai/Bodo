# `pd.Series.is_monotonic_decreasing`

`pandas.Series.is_monotonic_decreasing

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.is_monotonic_decreasing
>>> S = pd.Series(np.arange(100))
>>> f(S)
False
```

