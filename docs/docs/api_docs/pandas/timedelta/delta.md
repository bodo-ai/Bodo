# `pd.Timedelta.delta`

`pandas.Timedelta.delta`

### Example Usage

```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta(microseconds=23).delta
>>> f()
23000
```
