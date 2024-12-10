# `pd.Series.str.isupper`

`pandas.Series.str.isupper()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.isupper()
>>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0     True
1    False
2    False
3    False
4    False
5    False
6    False
dtype: boolean
```
