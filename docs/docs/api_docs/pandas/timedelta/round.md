# `pd.Timedelta.round`
                          
`pandas.Timedelta.round`

### Supported Arguments

- `freq`: String

### Example Usage
```py
>>> @bodo.jit
... def f():
...   return (pd.Timedelta(days=10, hours=12).round("D"), pd.Timedelta(days=10, hours=13).round("D"))
>>> f()
(Timedelta('10 days 00:00:00'), Timedelta('11 days 00:00:00'))
```

