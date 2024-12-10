# `pd.Series.cummin`

`pandas.Series.cummin(axis=None, skipna=True)`

### Supported Arguments None

!!! note
\- Series type must be numeric
\- Bodo does not accept any additional arguments for Numpy
compatibility

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.cummin()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0     0
1     0
2     0
3     0
4     0
     ..
95    0
96    0
97    0
98    0
99    0
Length: 100, dtype: int64
```
