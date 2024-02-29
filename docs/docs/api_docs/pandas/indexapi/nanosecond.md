# `pd.DateTimeIndex.nanosecond`

`pandasDatetimeIndex.nanosecond`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.nanosecond

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 01:01:01.0000001", end="2019-12-31 01:01:01.0000002", periods=5))
>>> f(I)
Int64Index([100, 125, 150, 175, 200], dtype='int64')
```




