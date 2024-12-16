# `pd.Series.rdiv`

`pandas.Series.rdiv(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument     | datatypes                                                                                                  |
|--------------|------------------------------------------------------------------------------------------------------------|
| `other`      | <ul><li>   numeric scalar </li><li> array with numeric data </li><li>  Series with numeric data </li></ul> |
| `fill_value` | numeric scalar                                                                                             |

!!! note
    `Series.rdiv` is only supported on Series of numeric data.


### Example Usage

``` py
>>> @bodo.jit
... def f(S, other):
...   return S.rdiv(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      1000.000000
1       499.500000
2       332.666667
3       249.250000
4       199.200000
          ...
995       0.005020
996       0.004012
997       0.003006
998       0.002002
999       0.001000
Length: 1000, dtype: float64
```

