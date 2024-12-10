# `pd.Timestamp.is_month_end`

`pandasTimestamp.is_month_end`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
...   ts2 = pd.Timestamp(year=2021, month=12, day=30)
...   return (ts1.is_month_end, ts2.is_month_end)
>>> f()
(True, False)
```
