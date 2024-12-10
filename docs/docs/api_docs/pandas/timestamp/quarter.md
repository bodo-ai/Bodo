# `pd.Timestamp.quarter`

`pandasTimestamp.quarter`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
...   ts2 = pd.Timestamp(year=2021, month=9, day=1)
...   return (ts1.quarter, ts2.quarter)
>>> f()
(4, 3)
```
