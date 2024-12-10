# `pd.Timestamp.is_month_start`

`pandasTimestamp.is_month_start`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
...   ts2 = pd.Timestamp(year=2021, month=12, day=2)
...   return (ts1.is_month_start, ts2.is_month_start)
>>> f()
(True, False)
```
