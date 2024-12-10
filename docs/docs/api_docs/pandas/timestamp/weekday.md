# `pd.Timestamp.weekday`


`pandasTimestamp.weekday()`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=9)
...   ts2 = pd.Timestamp(year=2021, month=12, day=10)
...   return (ts1.weekday(), ts2.weekday())
>>> f()
(3, 4)
```


