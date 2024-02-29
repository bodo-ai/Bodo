# `pd.Series.div`

`pandas.Series.div(other, level=None, fill_value=None, axis=0)`

### Supported Arguments
- 
| argument     | datatypes                                                                                                |
|--------------|----------------------------------------------------------------------------------------------------------|
| `other`      | <ul><li>  numeric scalar </li><li> array with numeric data </li><li> Series with numeric data </li></ul> |
| `fill_value` | numeric scalar                                                                                           |

!!! note
    `Series.div` is only supported on Series of numeric data.


### Example Usage

``` py
>>> @bodo.jit
... def f(S, other):
...   return S.div(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0         0.001000
1         0.002002
2         0.003006
3         0.004012
4         0.005020
          ...
995     199.200000
996     249.250000
997     332.666667
998     499.500000
999    1000.000000
Length: 1000, dtype: float64
```

