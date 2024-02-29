# `pd.Index.shape`


`pandasIndex.shape`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.shape

>>> I = pd.Index([1,2,3])
>>> f(I)
(3,)
```


