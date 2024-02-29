# `pd.Series.rpow`

`pandas.Series.rpow(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument                      | datatypes                                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| `other`                       | <ul><li>   numeric scalar </li><li> array with numeric data </li><li>  Series with numeric data </li></ul> |
| `fill_value`                  | numeric scalar                                                                                             |

!!! note
    `Series.rpow` is only supported on Series of numeric data.


### Example Usage

``` py
>>> @bodo.jit
... def f(S, other):
...   return S.rpow(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0                     1000
1                   998001
2                994011992
3             988053892081
4          980159361278976
              ...
995    3767675092665006833
996                      0
997   -5459658280481875879
998                      0
999                      1
Length: 1000, dtype: int64
```

