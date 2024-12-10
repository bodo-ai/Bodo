# `pd.Series.str.slice_replace`

`pandas.Series.str.slice_replace(start=None, stop=None, repl=None)`

### Supported Arguments

| argument                    | datatypes                            |
|-----------------------------|--------------------------------------|
| `start`                     |    Integer                           |
| `stop`                      |    Integer                           |
| `repl`                      |    String                            |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.slice_replace(1, 4)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0    A
1    c
2    1
3
4    @
5    a
6    #
dtype: object
```

