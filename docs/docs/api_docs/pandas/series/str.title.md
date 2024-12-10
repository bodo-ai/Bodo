# `pd.Series.str.title`

`pandas.Series.str.title()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.title()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0       A
1      Ce
2      14
3
4       @
5     A N
6    ^ Ef
dtype: object
```
