# `pd.Index.size`


`pandasIndex.size`


### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.size

>>> I = pd.Index([1,7,8,6])
>>> f(I)
4
```


