# `pd.Series.str.isnumeric`

`pandas.Series.str.isnumeric()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.isnumeric()
>>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0    False
1    False
2     True
3    False
4    False
5    False
6    False
dtype: boolean
```
