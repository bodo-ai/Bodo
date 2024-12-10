# `pd.Series.max`

`pandas.Series.max(axis=None, skipna=None, level=None, numeric_only=None)`

### Supported Arguments None

!!! note
\- Series type must be numeric
\- Bodo does not accept any additional arguments to pass to the
function

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.max()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
6
```
