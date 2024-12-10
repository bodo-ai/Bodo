# `pd.Timedelta.ceil`

`pandas.Timedelta.ceil(freq)`

### Supported Arguments

- `freq`: String

### Example Usage

```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).ceil("D")
>>> f()
11 days 00:00:00
```
