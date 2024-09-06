# `pd.Series.astype`

`pandas.Series.astype(dtype, copy=True, errors="raise", _bodo_nan_to_str=True)`


### Supported Arguments

| argument           | datatypes                                                                                                                                                        | other requirements                                                                                                                                                                                                                                     |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dtype`            | <ul><li>   String (string must be parsable by `np.dtype`) </li><li>  Valid type (see types)</li><li>   The following functions: float, int, bool, str </li></ul> | **Must be constant at   Compile Time**                                                                                                                                                                                                                 |
| `copy`             | Boolean                                                                                                                                                          | **Must be constant at Compile Time**                                                                                                                                                                                                                   |
| `_bodo_nan_to_str` | Boolean                                                                                                                                                          | <ul><li> **Must be constant at Compile Time** </li><li> Argument unique to  Bodo. When `True` NA values in when converting to string are represented as NA  instead of a string representation of the  NA value  '<NA>'), the default  Pandas behavior. |


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.astype(np.float32)
>>> S = pd.Series(np.arange(1000))
>>> f(S)
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
Length: 1000, dtype: float32
```
