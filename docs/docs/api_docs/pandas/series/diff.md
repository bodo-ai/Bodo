# `pd.Series.diff`

`pandas.Series.diff(periods=1)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `periods` | Integer |

!!! note
Bodo only supports numeric and datetime64 types

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.diff(3)
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0     NaN
1     NaN
2     NaN
3     3.0
4     3.0
     ...
95    3.0
96    3.0
97    3.0
98   -4.0
99   -4.0
Length: 100, dtype: float64
```
