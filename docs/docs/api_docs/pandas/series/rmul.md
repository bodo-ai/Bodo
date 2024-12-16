# `pd.Series.rmul`

`pandas.Series.rmul(other, level=None, fill_value=None, axis=0)`

### Supported Arguments

| argument                      | datatypes                                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| `other`                       | <ul><li>   numeric scalar </li><li> array with numeric data </li><li>  Series with numeric data </li></ul> |
| `fill_value`                  | numeric scalar                                                                                             |

!!! note
    `Series.rmul` is only supported on Series of numeric data.


### Example Usage

``` py
>>> @bodo.jit
... def f(S, other):
...   return S.rmul(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      1000
1      1998
2      2994
3      3988
4      4980
      ...
995    4980
996    3988
997    2994
998    1998
999    1000
Length: 1000, dtype: int64
```

