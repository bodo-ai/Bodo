# `pd.Index.sort_values`

`pandasIndex.sort_values(return_indexer=False, ascending=True, na_position="last", key=None)`


### Supported Arguments:

- `ascending`: can be True or False
- `na_position`: can be "first" or "last"

***Unsupported Index Types***

- IntervalIndex
- PeriodIndex
- MultiIndex


### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.sort_values()

>>> I = pd.Index([0, -1, 1, -5, 8, -13, -2, 3])
>>> f(I)
Int64Index([-13, -5, -2, -1, 0, 1, 3, 8], dtype='int64')
```

