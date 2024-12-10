# `pd.Timestamp.nanosecond`

`pandasTimestamp.nanosecond`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   ts2 = pd.Timestamp(12, unit="ns")
...   return ts2.nanosecond
>>> f()
12
```
