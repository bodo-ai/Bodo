# `pd.Timedelta.value`

`pandas.Timedelta.value`

### Example Usage

```py
>>> @bodo.jit
... def f():
...   return pd.Timedelta("13 nanoseconds").value
>>> f()
13
```
