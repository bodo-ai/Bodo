# `pd.Timedelta.to_numpy`
                          
`pandas.Timedelta.to_numpy()`

### Example Usage
```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_numpy()
>>> f()
871623013023000 nanoseconds
```

