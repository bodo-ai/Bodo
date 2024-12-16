# `pd.Index.argmin`

`pandasIndex.argmin(axis=None, skipna=True, *args, **kwargs)`

### Supported Arguments: None

***Unsupported Index Types***

  - IntervalIndex
  - MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.argmin()

>>> I = pd.Index([1, 4, 9, 0, 3])
>>> f(I)
3
```

