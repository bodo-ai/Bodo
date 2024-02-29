# `pd.Series.radd`

`pandas.Series.radd(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument     | datatypes                                                                                                  |
|--------------|------------------------------------------------------------------------------------------------------------|
| `other`      | <ul><li>   numeric scalar </li><li> array with numeric data </li><li>  Series with numeric data </li></ul> |
| `fill_value` | numeric scalar                                                                                             |

!!! note
    `Series.radd` is only supported on Series of numeric data.


### Example Usage

``` py
>>> @bodo.jit
... def f(S, other):
...   return S.radd(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      1001
1      1001
2      1001
3      1001
4      1001
      ...
995    1001
996    1001
997    1001
998    1001
999    1001
Length: 1000, dtype: int64
```

