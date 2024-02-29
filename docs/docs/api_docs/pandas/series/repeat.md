# `pd.Series.repeat`

`pandas.Series.repeat(repeats, axis=None)`

### Supported Arguments

| argument  | datatypes                                                                                      |
|-----------|------------------------------------------------------------------------------------------------|
| `repeats` | <ul><li>   Integer </li><li>   Array-like of integers the same length as the Series </li></ul> |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.repeat(3)
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      0
0      0
0      0
1      1
1      1
      ..
98    98
98    98
99    99
99    99
99    99
Length: 300, dtype: int64
```

### Combining / comparing / joining / merging

