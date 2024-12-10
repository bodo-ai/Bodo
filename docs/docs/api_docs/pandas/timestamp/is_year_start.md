# `pd.Timestamp.is_year_start`

`pandasTimestamp.is_year_start`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
...   return (ts1.is_year_start, ts2.is_year_start)
>>> f()
(False, True)
```
