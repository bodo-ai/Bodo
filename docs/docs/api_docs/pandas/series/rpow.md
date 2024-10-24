# `pd.Series.rpow`

`pandas.Series.rpow(other, level=None, fill_value=None, axis=0)`

### Argument Restrictions:
 * `other`: must be a numeric scalar or Series, Index, Array, List, or Tuple with numeric data.
 * `level`: only supports default value `None`.
 * `fill_value`: (optional, defaults to `None`) must be `Integer`, `Float` or `Boolean`.
 * `axis`: only supports default value `0`.

!!! note
	Input must be a Series of `Integer` or `Float` data.

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

