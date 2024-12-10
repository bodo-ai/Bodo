# `pd.DataFrame.pipe`

- pandas.DataFrame.pipe(func, \*args, \*\*kwargs)

### Supported Arguments

- `func`: JIT function or callable defined within a JIT function.
  - Additional arguments for `func` can be passed as additional arguments.

!!! note

```
`func` cannot be a tuple
```

### Example Usage

```py

>>> @bodo.jit
... def f():
...   def g(df, axis):
...       return df.max(axis)
...   df = pd.DataFrame({"A": [10,100,1000,10000]})
...   return df.pipe(g, axis=0)
...
>>> f()
A    10000
dtype: int64
```
