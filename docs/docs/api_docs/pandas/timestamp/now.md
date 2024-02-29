# `pd.Timestamp.now`


`pandasTimestamp.now(tz=None)`

### Supported Arguments:
- `tz`: constant string, integer, or None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   return pd.Timestamp.now()
>>> f()
Timestamp('2021-12-10 10:54:06.457168')

```
