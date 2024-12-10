# `pd.Series.str.istitle`

`pandas.Series.str.istitle()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.istitle()
>>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0     True
1    False
2    False
3    False
4    False
5    False
6     True
dtype: boolean
```
