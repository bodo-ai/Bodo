# `pd.Series.str.removeprefix`

`pandas.Series.str.removeprefix(prefix)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `prefix`                    |    String                              |

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.removeprefix("ab")
>>> S = pd.Series(["a", "ab", "abc", " abcd", "a bcd", "abcd", "xab"])
>>> f(S)
0        a
1         
2        c
3     abcd
4    a bcd
5       cd
6      xab
dtype: string
```

