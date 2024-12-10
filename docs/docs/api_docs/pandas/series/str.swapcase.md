# `pd.Series.str.swapcase`

`pandas.Series.str.swapcase()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.swapcase()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0       a
1      CE
2      14
3
4       @
5     A N
6    ^ Ef
dtype: object
```
