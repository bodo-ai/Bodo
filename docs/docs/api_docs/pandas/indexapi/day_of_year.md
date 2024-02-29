# `pd.DateTimeIndex.day_of_year`

`pandasDatetimeIndex.day_of_year`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.day_of_year

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([365, 365, 365, 1, 1], dtype='int64')
```

