# `pd.Index.copy`


`pandasIndex.copy(name=None, deep=False, dtype=None, names=None)`

***Unsupported Index Types***

- MultiIndex
- IntervalIndex

***Supported arguments***

- `name`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.copy(name="new_name")

>>> I = pd.Index([1,2,3], name = "origial_name")
>>> f(I)
Int64Index([1, 2, 3], dtype='int64', name='new_name')
```


