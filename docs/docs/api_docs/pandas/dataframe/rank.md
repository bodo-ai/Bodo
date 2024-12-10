# `pd.DataFrame.rank`

-  `pandas.DataFrame.rank(axis=0, method='average', numeric_only=NoDefault.no_default, na_option='keep', ascending=True, pct=False)`

### Supported Arguments

+-----------------------------+------------------------------------------------------------+
| argument                    | datatypes                                                  |
| `method`                    | -   String in {'average', 'min', 'max', 'first', 'dense'}  |
+-----------------------------+------------------------------------------------------------+
| `na_option`                 | -   String in {'keep', 'top', 'bottom'}                    |
+-----------------------------+------------------------------------------------------------+
| `ascending`                 | -   Boolean                                                |
+-----------------------------+------------------------------------------------------------+
| `pct`                       | -   Boolean                                                |
+-----------------------------+------------------------------------------------------------+

!!! note
    - Using `method='first'`  with `ascending=False` is currently unsupported.

### Example Usage

``` py
>>> @bodo.jit
... def f(df):
...     return df.rank(method='dense', na_option='keep', pct=True)
>>> df = pd.DataFrame('A': [np.nan, 4, 2, 4, 8, np.nan])
>>> f(df)
    A    B
0  NaN  0.5
1  1.0  1.0
2  0.5  1.0
3  1.0  NaN
```

