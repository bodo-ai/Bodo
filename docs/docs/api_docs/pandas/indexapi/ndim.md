# `pd.Index.ndim`

`pandasIndex.ndim`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.ndim

>>> I = pd.Index([1,2,3,4])
>>> f(I)
1
```
