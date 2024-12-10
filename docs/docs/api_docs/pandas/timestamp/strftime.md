# `pd.Timestamp.strftime`


`pandasTimestamp.strftime(format)`

### Supported Arguments

- `format`: string

### Example Usage

```py

>>> @bodo.jit
... def f():
...   return pd.Timestamp(year=2021, month=12, day=9, hour = 12).strftime('%Y-%m-%d %X')
>>> f()
'2021-12-09 12:00:00'
```


