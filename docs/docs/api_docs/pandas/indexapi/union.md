# `pd.Index.union`

`pandasIndex.union(other, sort=None)`

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
Bodo diverges from the Pandas API for Index.union() in several ways: duplicates are removed, the order of elements may be different, the shortcuts for returning the same Index are removed, and a NumericIndex is always returned instead of a RangeIndex.

### Example Usage

```py
>>> @bodo.jit(distributed=["I", "J"])
... def f(I, J):
...    return I.union(J)

>>> I = pd.Index([1, 2, 3, 4, 5])
>>> J = pd.Index([2, 4, 6, 8, 10, 12])
>>> f(I, J)
Int64Index([1, 2, 3, 4, 5, 6, 8, 10, 12], dtype='int64')
```
