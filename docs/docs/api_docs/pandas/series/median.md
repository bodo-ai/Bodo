# `pd.Series.median`

`pandas.Series.median(axis=None, skipna=None, level=None, numeric_only=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `skipna` | Boolean |

!!! note
\- Series type must be numeric
\- Bodo does not accept any additional arguments to pass to the
function

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.median()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
3.0
```
