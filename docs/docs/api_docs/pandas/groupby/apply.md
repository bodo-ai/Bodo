# `pd.core.groupby.Groupby.apply`

`pandas.core.groupby.Groupby.apply(func, *args, **kwargs)`

### Supported Arguments

- `func`: JIT function, callable defined within a JIT function that returns a DataFrame or Series
- Additional arguments for `func` can be passed as additional arguments.

### Example Usage

```py

>>> @bodo.jit
... def f(df, y):
...     return df.groupby("B", dropna=True).apply(lambda group, y: group.sum(axis=1) + y, y=y)
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> y = 4
>>> f(df, y)

B
421  0          6.510
   2          8.421
   4     233260.000
   6         16.210
   8          6.510
   10         8.421
   12    233260.000
   14        16.210
   16         6.510
   18         8.421
f31  1     233260.000
   3         16.210
   5          6.510
   7          8.421
   9     233260.000
   11        16.210
   13         6.510
   15         8.421
   17    233260.000
   19        16.210
dtype: float64
```
