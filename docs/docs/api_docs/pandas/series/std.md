# `pd.Series.std`

`pandas.Series.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|-------------------------------------|
| `skipna` | Boolean |
| `ddof` | Integer |

!!! note
\- Series type must be numeric
\- Bodo does not accept any additional arguments to pass to the
function

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.std()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
2.021975231891785
```
