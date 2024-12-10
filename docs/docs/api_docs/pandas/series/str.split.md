# `pd.Series.str.split`

`pandas.Series.str.split(pat=None, n=-1, expand=False)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|---------------------------------------|
| `pat` | String |
| `n` | Integer |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.split(" ")
>>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
>>> f(S)
0        [A]
1       [ce]
2       [14]
3       [, ]
4        [@]
5     [a, n]
6    [#, Ef]
dtype: object
```

```py
>>> @bodo.jit
... def f(S):
...     return S.str.split(" ", n=2)
>>> S = pd.Series(["alphabet soup is delicious", "hello world", "oh what a beautiful morning"])
>>> f(S)
0     [alphabet, soup, is delicious]
1                     [hello, world]
2    [oh, what, a beautiful morning]
dtype: object
```

```py
>>> @bodo.jit
... def f(S):
...     return S.str.split(n=1)
>>> S = pd.Series(["alphabet soup is delicious", "hello \n\tworld", "oh what a beautiful morning"])
>>> f(S)
0     [alphabet, soup is delicious]
1                     [hello, world]
2    [oh, what a beautiful morning]
dtype: object
```
