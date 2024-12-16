# `pd.Series.shift`

`pandas.Series.shift(periods=1, freq=None, axis=0, fill_value=None)`

### Supported Arguments

| argument  | datatypes |
|-----------|-----------|
| `periods` | Integer   |

!!! note
    This data type for the series must be one of:
    -   Integer
    -   Float
    -   Boolean
    -   datetime.data
    -   datetime64
    -   timedelta64
    -   string


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.shift(1)
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      NaN
1      0.0
2      1.0
3      2.0
4      3.0
      ...
95    94.0
96    95.0
97    96.0
98    97.0
99    98.0
Length: 100, dtype: float64
```

### Datetime properties

