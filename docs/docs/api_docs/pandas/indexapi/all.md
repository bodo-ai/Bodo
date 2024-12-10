# `pd.Index.all`

`pandasIndex.all(*args, **kwargs)`

### Supported Arguments: None

***Supported Index Types***

- NumericIndex (only Integers or Booleans)
- RangeIndex
- StringIndex
- BinaryIndex

!!! info "Important"
Bodo diverges from the Pandas API for StringIndex and BinaryIndex by always returning a boolean instead of sometimes returning a string.

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.all()

>>> I = pd.Index([1, 4, 9, 0, 3])
>>> f(I)
False
```
