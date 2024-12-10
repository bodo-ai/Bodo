# `pd.Index.is_floating`

`pandasIndex.is_floating()`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.is_floating()

>>> I = pd.Index([1, 2, 3])
>>> f(I)
False
```
