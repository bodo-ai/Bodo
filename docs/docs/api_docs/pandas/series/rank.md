# `pd.Series.rank`

`pandas.Series.rank(axis=0, method='average', numeric_only=NoDefault.no_default, na_option='keep', ascending=True, pct=False)`

### Supported Arguments

| argument    | datatypes                                             |
|-------------|-------------------------------------------------------|
| `method`    | String in {'average', 'min', 'max', 'first', 'dense'} |
| `na_option` | String in {'keep', 'top', 'bottom'}                   |
| `ascending` | Boolean                                               |
| `pct`       | Boolean                                               |

!!! note
    - Using `method='first'`  with `ascending=False` is currently unsupported.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.rank(method='dense', na_option='bottom', pct=True)
>>> S = pd.Series([np.nan, 4, 2, 4, 8, np.nan])
>>> f(S)
0    1.00
1    0.50
2    0.25
3    0.50
4    0.75
5    1.00
dtype: float64
```

