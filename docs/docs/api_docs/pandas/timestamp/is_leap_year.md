# `pd.Timestamp.is_leap_year`

`pandasTimestamp.is_leap_year`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2020, month=2,day=2)
...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
...   return (ts1.is_leap_year, ts2.is_leap_year)
>>> f()
(True, False)
```
