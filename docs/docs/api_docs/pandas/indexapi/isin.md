# `pd.Index.isin`

`pandasIndex.isin(values)`

### Supported Arguments

- `values`: list-like or array-like of values

***Unsupported Index Types***

  - MultiIndex
  - IntervalIndex
  - PeriodIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.isin([0, 2, 4])

>>> I = pd.Index([2, 4, 3, 4, 0, 3, 3, 5])
>>> f(I)
array([ True,  True, False,  True,  True, False, False, False])
```

