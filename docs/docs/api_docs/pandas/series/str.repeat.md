# `pd.Series.str.repeat`

`pandas.Series.str.repeat(repeats)`

### Supported Arguments

| argument | datatypes | other requirements |
|-----------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `repeats` | <ul><li> Integer </li><li> Array Like containing integers </li></ul> | If `repeats` is array like, then it must be the same length as the Series. |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.repeat(2)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0          AA
1        cece
2        1414
3
4          @@
5      a na n
6    ^ Ef^ Ef
dtype: object
```
