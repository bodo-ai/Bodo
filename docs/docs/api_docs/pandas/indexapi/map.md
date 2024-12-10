# `pd.Index.map`

`pandasIndex.map(mapper, na_action=None)`

***Unsupported Index Types***

- MultiIndex
- IntervalIndex

### Supported Arguments

- `mapper`: must be a function, function cannot return tuple type

### Example Usage

```py 
>>> @bodo.jit
... def f(I):
...   return I.map(lambda x: x + 2)

>>> I = pd.Index([1,None,3])
>>> f(I)
Float64Index([3.0, nan, 5.0], dtype='float64')
```

