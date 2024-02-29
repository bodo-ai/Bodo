# `pd.Timedelta.floor`
                          

`pandas.Timedelta.floor`

### Supported Arguments


- `freq`: String

### Example Usage
```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).floor("D")
>>> f()
10 days 00:00:00
```

