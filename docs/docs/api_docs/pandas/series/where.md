# `pd.Series.where`

`pandas.Series.where(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=NoDefault.no_default)`

### Supported Arguments

| argument | datatypes                                                          |
|----------|--------------------------------------------------------------------|
| `cond`   | <ul><li>  boolean array </li><li>   1d bool numpy array </li></ul> |
| `other`  | <ul><li>   1d numpy array </li><li> scalar   </li></ul>            |

!!! note
    Series can contain categorical data if `other` is a scalar


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.where((S % 3) != 0, 0)
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      0
1      1
2      2
3      0
4      4
      ..
95    95
96     0
97    97
98    98
99     0
Length: 100, dtype: int64
```

