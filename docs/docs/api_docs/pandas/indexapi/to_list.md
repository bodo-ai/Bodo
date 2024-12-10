# `pd.Index.to_list`

`pandasIndex.to_list()`

***Unsupported Index Types***

- PeriodIndex
- IntervalIndex
- MultiIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.to_list()

>>> I = pd.RangeIndex(5, -1, -1)
>>> f(I)
[5, 4, 3, 2, 1, 0]
```
