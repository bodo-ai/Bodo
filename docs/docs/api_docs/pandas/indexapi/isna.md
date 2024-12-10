# `pd.Index.isna`

`pandasIndex.isna()`

***Unsupported Index Types***

  - MultiIndex
  - IntervalIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.isna()

>>> I = pd.Index([1,None,3])
>>> f(I)
[False  True False]
```

