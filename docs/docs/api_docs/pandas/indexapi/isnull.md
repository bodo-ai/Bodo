# `pd.Index.isnull`

`pandasIndex.isnull()`

***Unsupported Index Types***

  - MultiIndex
  - IntervalIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.isnull()

>>> I = pd.Index([1,None,3])
>>> f(I)
[False  True False]
```


### Conversion

