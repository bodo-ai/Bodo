# `pd.DateTimeIndex.week`

`pandasDatetimeIndex.week`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.week

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([1, 1, 1, 1, 1], dtype='int64')
```

