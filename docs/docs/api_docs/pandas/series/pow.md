# `pd.Series.pow`

`pandas.Series.pow(other, level=None, fill_value=None, axis=0)`

### Argument Restrictions:
 * `other`: must be a numeric scalar or Series, Index, Array, List, or Tuple with numeric data.
 * `level`: only supports default value `None`.
 * `fill_value`: must be `Integer`, `Float`, `Boolean`, or `None`.
 * `axis`: only supports default value `0`.

!!! note
	Input must be a Series of `Integer` or `Float` data.

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

