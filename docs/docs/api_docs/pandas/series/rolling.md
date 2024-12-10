# `pd.Series.rolling`

`pandas.Series.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')`

### Supported Arguments

| argument | datatypes |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| `window` | <ul><li> Integer </li><li> String representing a Time Offset </li><li> Timedelta </li></ul> |
| `min_periods` | Integer |
| `center` | Boolean |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.rolling(2).mean()
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      NaN
  1      0.5
  2      1.5
  3      2.5
  4      3.5
        ...
  95    94.5
  96    95.5
  97    96.5
  98    97.5
  99    98.5
  Length: 100, dtype: float64
```
