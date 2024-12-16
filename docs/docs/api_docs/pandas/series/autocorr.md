# `pd.Series.autocorr`

`pandas.Series.autocorr(lag=1)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `lag`                       |    Integer                             |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.autocorr(3)
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
-0.49872171657407155
```

