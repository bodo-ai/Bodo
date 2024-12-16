# `pd.Series.str.pad`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.pad.html#pandas.Series.str.pad)

`pandas.Series.str.pad(width, side='left', fillchar=' ')`

### Argument Restrictions:
 * `width`: must be type `Integer`.
 * `side`: must be a compile time constant and must be `"left"`, `"right"` or `"both"`.
 * `fillchar`: must be type `Character`.

!!! note
	Input must be a Series of `String` data.

### Example Usage
``` py
>>> @bodo.jit
... def f(S):
...     return S.str.pad(5)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0        A
1       ce
2       14
3
4        @
5      a n
6     ^ Ef
dtype: object
```

