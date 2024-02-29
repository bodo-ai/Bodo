# `pd.Timestamp.week`


`pandasTimestamp.week`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
...   return (ts1.week, ts2.week)
>>> f()
(35, 38)
```


