# `pd.Series.last`

`pandas.Series.last(offset)`

### Supported Arguments

| argument | datatypes | other requirements |
|-------------------|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `offset` | - String or Offset type | - String argument be a valid [frequency alias](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases) |

!!! note
Series must have a valid DatetimeIndex and is assumed to already be
sorted. This function have undefined behavior if the DatetimeIndex
is not sorted.

### Example Usage

```py
>>> @bodo.jit
... def f(S, offset):
...     return S.last(offset)
>>> S = pd.Series(np.arange(100), index=pd.date_range(start='1/1/2022', end='12/31/2024', periods=100))
>>> f(S, "2M")
2024-11-05 16:43:38.181818176    94
2024-11-16 18:10:54.545454544    95
2024-11-27 19:38:10.909090912    96
2024-12-08 21:05:27.272727264    97
2024-12-19 22:32:43.636363632    98
2024-12-31 00:00:00.000000000    99
dtype: int64
```
