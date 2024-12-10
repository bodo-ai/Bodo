# `pd.Timestamp.is_quarter_start`

`pandasTimestamp.is_quarter_start`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
...   return (ts1.is_quarter_start, ts2.is_quarter_start)
>>> f()
(False, True)
```
