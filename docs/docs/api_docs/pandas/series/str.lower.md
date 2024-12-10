# `pd.Series.str.lower`

`pandas.Series.str.lower()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.lower()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0       a
1      ce
2      14
3
4       @
5     a n
6    ^ Ef
dtype: object
```
