# `pd.Series.mean`

`pandas.Series.mean(axis=None, skipna=None, level=None, numeric_only=None)`

### Supported Arguments None

!!! note
    - Series type must be numeric
    - Bodo does not accept any additional arguments to pass to the
    function

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.mean()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
2.95
```

