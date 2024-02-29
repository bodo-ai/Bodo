# `pd.Series.any`

`pandas.Series.any(axis=0, bool_only=None, skipna=True, level=None)`

### Supported Arguments None

!!! note
    Bodo does not accept any additional arguments for Numpy
    compatibility


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.any()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
True
```

