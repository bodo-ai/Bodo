# `pd.RangeIndex`

`pandas.RangeIndex(start=None, stop=None, step=None, dtype=None, copy=False, name=None)`

### Supported Arguments

- `start`: integer
- `stop`: integer
- `step`: integer
- `name`: String

### Example Usage

```py
>>> @bodo.jit
... def f():
...   return pd.RangeIndex(0, 10, 2)

>>> f(I)
RangeIndex(start=0, stop=10, step=2)
```
