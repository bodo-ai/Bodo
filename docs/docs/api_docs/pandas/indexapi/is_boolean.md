# `pd.Index.is_boolean`

`pandasIndex.is_boolean()`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.is_boolean()

>>> I = pd.Index([1, 2, 3])
>>> f(I)
False
```

