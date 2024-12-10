# `pd.Index.symmetric_difference`

`pandasIndex.symmetric_difference(other, sort=None)`

### Supported Arguments:

  - `other`: can be an Index, Series, or 1-dim numpy array with a matching type for the Index

***Supported Index Types***

  - NumericIndex
  - StringIndex
  - BinaryIndex
  - RangeIndex
  - DatetimeIndex
  - TimedeltaIndex

!!! info "Important"
      Bodo diverges from the Pandas API for Index.symmetric_difference() in several ways: the order of elements may be different and a NumericIndex is always returned instead of a RangeIndex.

### Example Usage

```py
>>> @bodo.jit(distributed=["I", "J"])
... def f(I, J):
...    return I.difference(J)

>>> I = pd.Index([1, 2, 3, 4, 5])
>>> J = pd.Index([2, 4, 6, 8, 10, 12])
>>> f(I, J)
Int64Index([1, 3, 5, 6, 8, 10, 12], dtype='int64')
```

