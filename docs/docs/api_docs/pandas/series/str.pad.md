# `pd.Series.str.pad`

`pandas.Series.str.pad(width, side='left', fillchar=' ')`

### Supported Arguments

| argument   | datatypes                          | other requirements                   |
|------------|------------------------------------|--------------------------------------|
| `width`    | Integer                            |                                      |
| `width`    | One of ("left",  "right",  "both") | **Must be constant at Compile Time** |
| `fillchar` | String with a single character     |                                      |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.pad(5)
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0        A
1       ce
2       14
3
4        @
5      a n
6     ^ Ef
dtype: object
```

