# `pd.Timestamp.value`


`pandasTimestamp.value`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   return pd.Timestamp(12345, unit="ns").value
>>> f()
12345
```


