# `pd.Series.pipe`

-  pandas.Series.pipe(func, *args, **kwargs)

### Supported Arguments

| argument | datatypes                                               | other requirements                                                     |
|----------|---------------------------------------------------------|------------------------------------------------------------------------|
| `func`   | JIT function or callable defined within a JIT function. | Additional arguments for `func` can be passed as additional arguments. |

!!! note
    `func` cannot be a tuple


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     def g(row, y):
...         return row + y
...
...     def f(row):
...         return row * 2
...
...     return S.pipe(h).pipe(g, y=32)
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      32
1      34
2      36
3      38
4      40
     ...
95    222
96    224
97    226
98    228
99    230
Length: 100, dtype: int64
```

### Computations / Descriptive Stats

Statistical functions below are supported without optional arguments
unless support is explicitly mentioned.

