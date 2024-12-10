# `pd.Index.empty`

`pandasIndex.empty`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.empty

>>> I = pd.Index(["A", "B", "C"])
>>> f(I)
False
```
