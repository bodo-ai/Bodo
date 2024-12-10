# `pd.Index.inferred_type`

`pandasIndex.inferred_type`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.dtype

>>> I = pd.Index(["A", "E", "I", "O", "U"])
>>> f(I)
'string'
```
