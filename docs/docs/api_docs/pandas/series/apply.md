# `pd.Series.apply`

- pandas.Series.applyf(func, convert_dtype=True, args=(), \*\*kwargs)

### Supported Arguments

| argument | datatypes | other requirements |
|----------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `func` | Additional arguments for `func` can be passed as additional arguments. | <ul><li> JIT function or callable defined within a JIT function </li><li> Numpy ufunc </li><li> Constant String which is the name of a supported Series method or Numpy ufunc </li> |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...   return S.apply(lambda x: x ** 0.75)
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
