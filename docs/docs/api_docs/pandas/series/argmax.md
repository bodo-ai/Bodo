# `pd.Series.argmax`

`pandas.Series.argmax(axis=None, skipna=True)`

### Supported Arguments None

!!! note
    - Series dtype must be of type `int`, `float`, `bool` or `datetime`
    - Bodo only accepts default values for `axis` and `skipna`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.argmax()
>>> S = pd.Series([4, -2, 3, 6, -1])
>>> f(S)
3
```