# `pd.Index.is_numeric`

`pandasIndex.is_numeric()`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.is_numeric()

>>> I = pd.Index([1, 2, 3])
>>> f(I)
True
```
