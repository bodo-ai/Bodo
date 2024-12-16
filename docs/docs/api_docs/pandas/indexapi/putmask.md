# `pd.Index.putmask`

`pandasIndex.putmask(cond, other=None)`

### Supported Arguments:

  - `cond`: can be a Series or 1-dim array of booleans
  - `other`: can be a scalar, non-categorical Series, 1-dim numpy array or StringArray with a matching type for the Index

***Unsupported Index Types***

  - IntervalIndex
  - MultiIndex

!!! info "Important"
      Only supported for CategoricalIndex if the elements of other are the same as (or a subset of) the categories of the CategoricalIndex.

### Example Usage

```py
>>> @bodo.jit
... def f(I, C, O):
...   return I.putmask(C, O)

>>> I = pd.Index(["A", "B", "C", "D", "E"])
>>> C = pd.array([True, False, True, True, False])
>>> O = pd.Series(["a", "e", "i", "o", "u")
>>> f(I, C, O)
Index(['a', 'B', 'i', 'o', 'E'], dtype='object')
```


