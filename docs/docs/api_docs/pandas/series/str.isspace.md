# `pd.Series.str.isspace`

`pandas.Series.str.isspace()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.isspace()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0    False
1    False
2    False
3     True
4    False
5    False
6    False
dtype: boolean
```
