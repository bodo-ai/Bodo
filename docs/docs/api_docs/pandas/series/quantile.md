# `pd.Series.quantile`

`pandas.Series.quantile(q=0.5, interpolation='linear')`

### Supported Arguments

| argument                    | datatypes                                                                            |
|-----------------------------|--------------------------------------------------------------------------------------|
| `q`                         | <ul><li>   Float in [0.0, 1.0] </li><li>  Iterable of floats in [0.0, 1.0] </li></ul |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.quantile([0.25, 0.5, 0.75])
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0.25    1.0
0.50    3.0
0.75    5.0
dtype: float64
```

