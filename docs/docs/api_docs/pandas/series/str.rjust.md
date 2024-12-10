# `pd.Series.str.rjist`

`pandas.Series.str.rjust(width, fillchar=' ')`

Supported arguments\`:

| argument | datatypes |
|------------|--------------------------------|
| `width` | Integer |
| `fillchar` | String with a single character |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.rjust(10)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0             A
1            ce
2            14
3
4             @
5           a n
6          ^ Ef
dtype: object
```
