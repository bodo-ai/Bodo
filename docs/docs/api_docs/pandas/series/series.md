# `pd.Series`

-   `pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)`

### Supported Arguments

| argument | datatypes                                                                                                               | other requirements                                                                                                                               |
|----------|-------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `data`   | <ul><li> Series type</li> <li>List type</li><li>  Array type <li><li>  Constant  Dictionary  </li><li>  None </li></ul> |                                                                                                                                                  |
| `index`  | SeriesType                                                                                                              |                                                                                                                                                  |
| `dtype`  | <ul><li>  Numpy or Pandas Type </li> <li>String name for Numpy/Pandas Type  </li></ul>                                  | <ul><li>  **Must be constant at Compile Time** </li><li> String/Data Type must be one of the  supported types (see `Series.astype()`) </li></ul> |
| `name`   | String                                                                                                                  |                                                                                                                                                  |

!!! note
    If `data` is a Series and `index` is provided, implicit alignment is
    not performed yet.

### Example Usage

``` py
>>> @bodo.jit
... def f():
...     return pd.Series(np.arange(1000), dtype=np.float64, name="my_series")
>>> f()
0        0.0
1        1.0
2        2.0
3        3.0
4        4.0
      ...
995    995.0
996    996.0
997    997.0
998    998.0
999    999.0
Name: my_series, Length: 1000, dtype: float64
```

### Attributes

