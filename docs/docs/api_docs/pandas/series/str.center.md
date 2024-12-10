# `pd.Series.str.center`

`pandas.Series.str.center(width, fillchar=' ')`

### Supported Arguments

| argument | datatypes |
|------------|--------------------------------|
| `width` | Integer |
| `fillchar` | String with a single character |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.str.center(4)
>>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
>>> f(S)
0     a
1     ce
2    Erw
3     a3
4     @
5    a n
6    ^ Ef
dtype: object
```
