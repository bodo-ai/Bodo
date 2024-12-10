# `pd.core.groupby.Groupby.rolling`

`pandas.core.groupby.Groupby.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')`

### Supported Arguments

- `window`: Integer, String, Datetime, Timedelta
- `min_periods`: Integer
- `center`: Boolean
- `on`: Column label
  - **Must be constant at Compile Time**

!!! note\
This is equivalent to performing the DataFrame API
on each groupby. All operations of the rolling API
can be used with groupby.

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").rolling(2).mean
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

           A            C
B
421 0    NaN          NaN
    2    NaN          NaN
    4   12.5  116616.7550
    6    NaN       7.3155
    8   12.5  116616.7550
    10   NaN       7.3155
    12  12.5  116616.7550
    14   NaN       7.3155
    16  12.5  116616.7550
    18   NaN       7.3155
f31 1   12.5  116616.7550
    3    NaN       7.3155
    5   12.5  116616.7550
    7    NaN       7.3155
    9   12.5  116616.7550
    11   NaN       7.3155
    13  12.5  116616.7550
    15   NaN       7.3155
    17  12.5  116616.7550
    19   NaN       7.3155
```
