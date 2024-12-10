# `pd.Float64Index`

`pandas.Float64Index(data=None, dtype=None, copy=False, name=None)`

### Supported Arguments

- `data`: list or array
- `copy`: Boolean
- `name`: String


### Example Usage
```py
>>> @bodo.jit
... def f():
... return pd.Float64Index(np.arange(3))

>>> f()
Float64Index([0.0, 1.0, 2.0], dtype='float64')
```

 
 
 
## DatetimeIndex

`DatetimeIndex` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.


