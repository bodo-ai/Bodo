# `pd.DateTimeIndex.year`

`pandasDatetimeIndex.year`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.year

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([2019, 2019, 2019, 2020, 2020], dtype='int64')
```
