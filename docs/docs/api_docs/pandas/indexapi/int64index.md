# `pd.Int64Index`

`pandas.Int64Index(data=None, dtype=None, copy=False, name=None)`

### Example Usage

```py
>>> @bodo.jit
... def f():
... return pd.Int64Index(np.arange(3))

>>> f()
Int64Index([0, 1, 2], dtype='int64')
```
