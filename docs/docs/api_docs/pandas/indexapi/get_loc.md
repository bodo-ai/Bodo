# `pd.Index.get_loc`

- `pandas.Index.get_loc(key, method=None, tolerance=None)`

!!! note
Should be about as fast as standard python, maybe slightly slower.

***Unsupported Index Types***

- CategoricalIndex
- MultiIndex
- IntervalIndex

### Supported Arguments

- `key`: must be of same type as the index

!!! info "Important"

```
- Only works for index with unique values (scalar return).
- Only works with replicated Index
```

### Example Usage

```py
   
>>> @bodo.jit
... def f(I):
...   return I.get_loc(2)

>>> I = pd.Index([1,2,3])
>>> f(I)
1
```
