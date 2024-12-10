# `pd.Timedelta.total_seconds`

`pandas.Timedelta.total_seconds()`

### Example Usage
```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).total_seconds()
>>> f()
871623.013023
```
