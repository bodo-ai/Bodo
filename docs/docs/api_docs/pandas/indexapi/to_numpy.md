# `pd.Index.to_numpy`

`pandasIndex.to_numpy(dtype=None, copy=True, na_value=None)`

### Supported Arguments:

- `copy`: can be a True or False

***Unsupported Index Types***

- PeriodIndex
- MultiIndex

!!! info "Important"
Sometimes Bodo returns a Pandas array instead of a np.ndarray. Cases
include a NumericIndex of integers containing nulls, or a CategoricalIndex.

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.numpy()

>>> I = pd.Index([1, 9, -1, 3, 0, 1, 6])
>>> f(I)
[ 1  9 -1  3  0  1  6]
```
