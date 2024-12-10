# `pd.Series.str.capitalize`

`pandas.Series.str.capitalize()`

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.str.capitalize()
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0       A
1      Ce
2     Erw
3      A3
4       @
5     A n
6    ^ Ef
dtype: object
```
