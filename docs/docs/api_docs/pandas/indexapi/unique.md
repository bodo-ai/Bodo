# `pd.Index.unique`

`pandasIndex.unique()`

***Unsupported Index Types***

- IntervalIndex
- PeriodIndex
- MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.unique()

>>> I = pd.Index([1, 5, 2, 1, 0, 1, 5, 2, 1, 3])
>>> f(I)
Int64Index([1, 5, 2, 0, 3], dtype='int64')
```
