# `pd.Timestamp.ceil`

`pandasTimestamp.ceil(freq, ambiguous='raise', nonexistent='raise')`

### Supported Arguments

- `freq`: string

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
...   return (ts1, ts2)
>>> f()
(Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-10 00:00:00'))
```
