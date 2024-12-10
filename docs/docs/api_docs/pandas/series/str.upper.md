# `pd.Series.str.upper`

`pandas.Series.str.upper()`

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.upper()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0       A
1      CE
2      14
3
4       @
5     A N
6    ^ Ef
dtype: object
```

