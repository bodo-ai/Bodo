# `pd.DateTimeIndex.hour`

`pandasDatetimeIndex.hour`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.hour

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([2, 12, 22, 9, 19], dtype='int64')
```
