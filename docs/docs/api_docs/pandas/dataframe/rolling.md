# `pd.DataFrame.rolling`

`pandas.DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')`

### Supported Arguments

- `window`: Integer, String (must be parsable as a time offset),`datetime.timedelta` ,pd.Timedelta\`, List/Tuple of column labels
- `min_periods`: Integer
- `center`: boolean
- `on`: Scalar column label
  - **Must be constant at Compile Time**
- `dropna`:boolean
  - **Must be constant at Compile Time**

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3,4,5]})
...   return df.rolling(3,center=True).mean()
>>> f()
     A
0  NaN
1  2.0
2  3.0
3  4.0
4  NaN
```

For more information, please see [the Window section][pd_window_section].
