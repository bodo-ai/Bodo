# `pd.Series.append`

`pandas.Series.append(to_append, ignore_index=False, verify_integrity=False)`

### Supported Arguments

| argument       | datatypes                                                                        | other requirements                   |
|----------------|----------------------------------------------------------------------------------|--------------------------------------|
| `to_append`    | <ul><li>  Series </li><li>  List of Series  </li><li> Tuple of Series </li></ul> |                                      |
| `ignore_index` | Boolean                                                                          | **Must be constant at Compile Time** |

!!! note
    Setting a name for the output Series is not supported yet


!!! info "Important"
    Bodo currently concatenates local data chunks for distributed
    datasets, which does not preserve global order of concatenated
    objects in output.
   

### Example Usage

``` py
>>> @bodo.jit
... def f(S1, S2):
...     return S1.append(S2)
>>> S = pd.Series(np.arange(100))
>>> f(S, S)
0      0
1      1
2      2
3      3
4      4
      ..
95    95
96    96
97    97
98    98
99    99
Length: 200, dtype: int64
```

### Time series-related

