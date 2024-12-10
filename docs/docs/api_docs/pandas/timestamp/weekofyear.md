# `pd.Timestamp.weekofyear`


`pandasTimestamp.weekofyear`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
...   return (ts1.weekofyear, ts2.weekofyear)
>>> f()
(35, 38)
```


