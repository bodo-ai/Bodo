# `pd.DataFrame.last`

`pandas.DataFrame.last(offset)`

### Supported Arguments

- `offset`: String or Offset type
  - String argument must be a valid [frequency alias](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases){target=blank}

!!! note
DataFrame must have a valid DatetimeIndex and is assumed to already be sorted.
This function have undefined behavior if the DatetimeIndex is not sorted.

### Example Usage

```py
>>> @bodo.jit
... def f(df, offset):
...     return df.last(offset)
>>> df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)}, index=pd.date_range(start='1/1/2022', end='12/31/2024', periods=100))
>>> f(df, "2M")
                              A    B
2024-11-05 16:43:38.181818176  94  194
2024-11-16 18:10:54.545454544  95  195
2024-11-27 19:38:10.909090912  96  196
2024-12-08 21:05:27.272727264  97  197
2024-12-19 22:32:43.636363632  98  198
2024-12-31 00:00:00.000000000  99  199
```
