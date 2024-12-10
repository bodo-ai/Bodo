# `pd.Series.combine`

`pandas.Series.combine(other, func, fill_value=None)`

### Supported Arguments

| argument        | datatypes                                                                   | other requirements        |
|-----------------|-----------------------------------------------------------------------------|---------------------------|
| `other`         | <ul><li>   Array  </li><li> Series  </li></ul>                              |                           |
| `func`          | -   Function that takes two scalar arguments and   returns a scalar  value. |                           |
| `fill_value`    | scalar                                                                      | Must be provided if the   |
|                 |                                                                             | Series lengths aren't     |
|                 |                                                                             | equal and the dtypes      |
|                 |                                                                             | aren't floats.            |

### Example Usage

``` py
>>> @bodo.jit
... def f(S, other):
...   return S.combine(other, lambda a, b: 2 * a + b)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
0      1002
1      1003
2      1004
3      1005
4      1006
      ...
995    1997
996    1998
997    1999
998    2000
999    2001
Length: 1000, dtype: int64
```

