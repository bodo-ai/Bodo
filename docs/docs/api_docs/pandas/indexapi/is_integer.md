# `pd.Index.is_integer`

`pandasIndex.is_integer()`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.is_integer()

>>> I = pd.Index([1, 2, 3])
>>> f(I)
True
```
