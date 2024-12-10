# `pd.Series.argmin`

`pandas.Series.argmin(axis=None, skipna=True)`

### Supported Arguments None

!!! note
    - Series dtype must be a numeric type
    - Bodo only accepts default values for `axis` and `skipna`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.argmin()
>>> S = pd.Series([4, -2, 3, 6, -1])
>>> f(S)
1
```