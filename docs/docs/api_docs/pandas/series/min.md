# `pd.Series.min`

`pandas.Series.min(axis=None, skipna=None, level=None, numeric_only=None)`

### Supported Arguments None

!!! note
    - Series type must be numeric
    - Bodo does not accept any additional arguments to pass to the
    function

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.min()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0
```

