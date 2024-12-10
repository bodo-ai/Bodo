# `pd.Series.pct_change`

`pandas.Series.pct_change(periods=1, fill_method='pad', limit=None, freq=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `periods` | Integer |

!!! note
\- Series type must be numeric
\- Bodo does not accept any additional arguments to pass to shift

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.pct_change(3)
>>> S = (pd.Series(np.arange(100)) % 7) + 1
>>> f(S)
0          NaN
1          NaN
2          NaN
3     3.000000
4     1.500000
        ...
95    1.500000
96    1.000000
97    0.750000
98   -0.800000
99   -0.666667
Length: 100, dtype: float64
```
