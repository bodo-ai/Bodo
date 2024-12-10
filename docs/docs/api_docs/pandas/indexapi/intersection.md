# `pd.Index.intersection`

`pandasIndex.intersection(other, sort=None)`

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
Bodo diverges from the Pandas API for Index.intersection() in several ways: the default is sort=None, and a NumericIndex is always returned instead of a RangeIndex.

### Example Usage

```py
>>> @bodo.jit(distributed=["I", "J"])
... def f(I, J):
...    return I.intersection(J)

>>> I = pd.Index([1, 2, 3, 4, 5])
>>> J = pd.Index([2, 4, 6, 8, 10, 12])
>>> f(I, J)
Int64Index([2, 4], dtype='int64')
```
