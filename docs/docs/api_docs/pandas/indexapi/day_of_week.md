# `pd.DateTimeIndex.day_of_week`

`pandasDatetimeIndex.day_of_week`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.day_of_week

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([1, 1, 1, 2, 2], dtype='int64')
```

