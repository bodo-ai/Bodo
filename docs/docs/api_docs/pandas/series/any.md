# `pd.Series.any`

[Link to external documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.any.html)

`pandas.Series.any(axis=0, bool_only=None, skipna=True, level=None)`

### Supported Arguments:
 * `axis`: only supports default value `0`.
 * `bool_only`: only supports default value `None`.
 * `skipna`: only supports default value `True`.
 * `level`: only supports default value `None`.

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

