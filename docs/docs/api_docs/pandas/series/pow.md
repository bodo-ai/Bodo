# `pd.Series.pow`

`pandas.Series.pow(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument                      | datatypes                                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| `other`                       | <ul><li>   numeric scalar </li><li> array with numeric data </li><li>  Series with numeric data </li></ul> |
| `fill_value`                  | numeric scalar                                                                                             |

!!! note
    `Series.pow` is only supported on Series of numeric data.


### Example Usage

``` py
>>> @bodo.jit
... def f(S, other):
...   return S.pow(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0                        1
1                        0
2     -5459658280481875879
3                        0
4      3767675092665006833
              ...
995        980159361278976
996           988053892081
997              994011992
998                 998001
999                   1000
Length: 1000, dtype: int64
```

