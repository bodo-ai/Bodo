# `pd.Series.cumprod`

`pandas.Series.cumprod(axis=None, skipna=True)`

### Supported Arguments None

!!! note
\- Series type must be numeric
\- Bodo does not accept any additional arguments for Numpy
compatibility

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.cumprod()
>>> S = (pd.Series(np.arange(10)) % 7) + 1
>>> f(S)
0        1
1        2
2        6
3       24
4      120
5      720
6     5040
7     5040
8    10080
9    30240
dtype: int64
```
