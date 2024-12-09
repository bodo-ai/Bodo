# `pd.Series.str.casefold`

`pandas.Series.str.casefold()`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.casefold()
>>> S = pd.Series(["A", "CE", "Erw", "A3", "@", "ß", "^ Ef"])
>>> f(S)
0       a
1      ce
2     erw
3      a3
4       @
5      ss
6    ^ ef
dtype: object
```