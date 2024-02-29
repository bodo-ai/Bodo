# `pd.Series.map`

`pandas.Series.map(arg, na_action=None)`

### Supported Arguments

| argument | datatypes                                                                                                                                                                                                             |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arg`    | <ul><li>   Dictionary   </li><li>   JIT function or callable defined within a JIT function </li><li>   Constant String which refers to a supported Series method or Numpy  ufunc  </li><li>   Numpy ufunc  </li></ul> |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...   return S.map(lambda x: x ** 0.75)
>>> S = pd.Series(np.arange(100))
>>> f(S)
0      0.000000
1      1.000000
2      1.681793
3      2.279507
4      2.828427
        ...
95    30.429352
96    30.669269
97    30.908562
98    31.147239
99    31.385308
Length: 100, dtype: float64
```

