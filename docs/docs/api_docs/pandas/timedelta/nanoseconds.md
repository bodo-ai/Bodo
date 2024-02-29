# `pd.Timedelta.nanoseconds`
                          

`pandas.Timedelta.nanoseconds`

### Example Usage
```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).nanoseconds
>>> f()
0
```

