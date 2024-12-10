# `pd.core.groupby.Groupby.pipe`

`pandas.core.groupby.Groupby.pipe(func, \*args, **kwargs)`

### Supported Arguments

- `func`: JIT function, callable defined within a JIT function.
    - Additional arguments for `func` can be passed as additional arguments.


!!! note
    `func` cannot be a tuple

### Example Usage

```py

>>> @bodo.jit
... def f(df, y):
...     return df.groupby("B").pipe(lambda grp, y: grp.sum() - y, y=y)
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> y = 5
>>> f(df, y)

         A            C
B
421  120.0  1166162.550
f31    5.0       68.155
```
  
