# `pd.Series.str.contains`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html#pandas.Series.str.contains)

`pandas.Series.str.contains(pat, case=True, flags=0, na=None, regex=True)`

### Argument Restrictions:
 * `pat`: must be type `String`.
 * `case`: must be a compile time constant and must be type `Boolean`.
 * `flags`: must be type `Integer`.
 * `na`: only supports default value `None`.
 * `regex`: must be a compile time constant and must be type `Boolean`.

!!! note
	Input must be a Series of `String` data.

!!! note
	Argument `na` has default value `None` that's different than Pandas default.

### Example Usage:
``` py
>>> @bodo.jit
... def f(S):
...     return S.str.contains("a.+")
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0    False
1    False
2    False
3     True
4    False
5     True
6    False
dtype: boolean
```

