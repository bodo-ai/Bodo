# `pd.Series.str.isalnum`

`pandas.Series.str.isalnum()`

```py
>>> @bodo.jit
... def f(S):
...     return S.str.isalnum()
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0     True
1     True
2     True
3    False
4    False
5    False
6    False
dtype: boolean
```
