# `pd.Timestamp.days_in_month`


`pandasTimestamp.days_in_month`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
...   return ts2.days_in_month
>>> f()
31
```

