# `pd.core.groupby.DataFrameGroupby.aggregate`

`pandas.core.groupby.DataFrameGroupby.aggregate(func, *args, **kwargs)`


### Supported Arguments
  
- `func`: JIT function, callable defined within a JIT function, constant dictionary mapping column name to a function
- Additional arguments for `func` can be passed as additional arguments.

!!! note
    - Passing a list of functions is also supported if only one output column is selected.
    - Output column names can be specified using keyword arguments and `pd.NamedAgg()`.

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B", dropna=True).agg({"A": lambda x: max(x)})
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

        A
B
421  24.0
f31   2.0
```
  
