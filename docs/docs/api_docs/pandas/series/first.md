# `pd.Series.first`

`pandas.Series.first(offset)`

### Supported Arguments

| argument | datatypes | other requirements |
|-------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `offset` | String or Offset type | String argument be a valid [frequency alias](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases) |

!!! note
Series must have a valid DatetimeIndex and is assumed to already be
sorted. This function have undefined behavior if the DatetimeIndex
is not sorted.

### Example Usage

```py
>>> @bodo.jit
... def f(S, offset):
...     return S.first(offset)
>>> S = pd.Series(np.arange(100), index=pd.date_range(start='1/1/2022', end='12/31/2024', periods=100))
>>> f(S, "2M")
2022-01-01 00:00:00.000000000    0
2022-01-12 01:27:16.363636363    1
2022-01-23 02:54:32.727272727    2
2022-02-03 04:21:49.090909091    3
2022-02-14 05:49:05.454545454    4
2022-02-25 07:16:21.818181818    5
dtype: int64
```
