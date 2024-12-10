# `pd.Timestamp.date`

`pandasTimestamp.date()`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).date()
...   return (ts1, ts2)
>>> f()
(Timestamp('2021-12-09 09:57:44.114123'), datetime.date(2021, 12, 9))
```
