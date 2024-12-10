# `pd.Index.nlevels`

`pandasIndex.nlevels`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.nlevels

>>> I = pd.MultiIndex.from_arrays([[1, 2, 3, 4],["A", "A", "B", "B"]])
>>> f(I)
2
```
