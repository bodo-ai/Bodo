# `pd.Series.any`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.any.html#pandas.Series.any)

`pandas.Series.any(axis=0, bool_only=None, skipna=True)`

### Argument Restrictions:

- `axis`: only supports default value `0`.
- `bool_only`: only supports default value `None`.
- `skipna`: only supports default value `True`.

!!! note
Argument `bool_only` has default value `None` that's different than Pandas default.

!!! note
Bodo does not accept any additional arguments for Numpy
compatibility

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.any()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
True
```
