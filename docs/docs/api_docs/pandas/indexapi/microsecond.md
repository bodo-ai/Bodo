# `pd.DateTimeIndex.microsecond`

`pandasDatetimeIndex.microsecond`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.microsecond

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 01:01:01", end="2019-12-31 01:01:02", periods=5))
>>> f(I)
Int64Index([0, 250000, 500000, 750000, 0], dtype='int64')
```

