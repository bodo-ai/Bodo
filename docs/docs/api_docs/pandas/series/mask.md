# `pd.Series.mask`

`pandas.Series.mask(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=NoDefault.no_default)`

### Supported Arguments

| argument | datatypes |
|----------|----------------------------------------------------------------------|
| `cond` | <ul><li> boolean array </li></li> 1d bool numpy array </li></ul> |
| `other` | <ul><li> 1d numpy array </li></li> scalar </li></ul> |

!!! note
Series can contain categorical data if `other` is a scalar

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.mask((S % 3) != 0, 0)
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      0
1      0
2      0
3      3
4      0
      ..
95     0
96    96
97     0
98     0
99    99
Length: 100, dtype: int64
```

### Missing data handling
