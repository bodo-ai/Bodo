# `pd.Timedelta.to_pytimedelta`
                            

`pandas.Timedelta.to_pytimedelta()`

### Example Usage
```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_pytimedelta()
>>> f()
10 days, 2:07:03.013023
```

