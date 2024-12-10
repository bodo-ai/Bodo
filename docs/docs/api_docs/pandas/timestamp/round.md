# `pd.Timestamp.round`

`pandasTimestamp.round(freq, ambiguous='raise', nonexistent='raise')`

### Supported Arguments

- `freq`: string

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 12).round()
...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 13).round()
...   return (ts1, ts2)
>>> f()
(Timestamp('2021-12-09 00:00:00'),Timestamp('2021-12-10 00:00:00'))
```
