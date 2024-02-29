# `pd.Index.is_interval`

`pandasIndex.is_interval()`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.is_interval()

>>> I = pd.Index([1, 2, 3])
>>> f(I)
False
```

