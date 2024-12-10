# `pd.Index.T`

`pandasIndex.T`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.T

>>> I = pd.Index(["A", "E", "I", "O", "U"])
>>> f(I)
Index(["A", "E", "I", "O", "U"], dtype='object')
```

### Type information
