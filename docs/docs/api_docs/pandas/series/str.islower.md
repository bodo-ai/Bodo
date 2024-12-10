# `pd.Series.str.islower`

`pandas.Series.str.islower()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.islower()
>>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0    False
1     True
2    False
3     True
4    False
5     True
6    False
dtype: boolean
```
