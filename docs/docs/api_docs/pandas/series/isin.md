# `pd.Series.isin`

`pandas.Series.isin(values)`

### Supported Arguments

| argument | datatypes                                                   |
|----------|-------------------------------------------------------------|
| `values` | <ul><li>   Series </li><li> Array </li><li> List </li></ul> |

!!! note
    `values` argument supports both distributed array/Series
    and replicated list/array/Series


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.isin([3, 11, 98])
>>> S = pd.Series(np.arange(100))
>>> f(S)
0     False
1     False
2     False
3      True
4     False
      ...
95    False
96    False
97    False
98     True
99    False
Length: 100, dtype: bool
```

