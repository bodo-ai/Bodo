# `pd.Timestamp.isoformat`

`pandasTimestamp.isoformat()`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
...   return (ts1, ts2)
>>> f()
'2021-12-09T09:57:44'
```
