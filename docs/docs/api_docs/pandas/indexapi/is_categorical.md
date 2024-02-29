# `pd.Index.is_categorical`

`pandasIndex.is_categorical()`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.is_categorical()

>>> I = pd.Index([1, 2, 3])
>>> f(I)
False
```

