# `pd.Series.str.replace`

`pandas.Series.str.replace(pat, repl, n=- 1, case=None, flags=0, regex=None)`

### Supported Arguments 

- `regex`

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.replace("(a|e)", "yellow")
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0           A
1     cyellow
2          14
3
4           @
5    yellow n
6        ^ Ef
dtype: object
```

