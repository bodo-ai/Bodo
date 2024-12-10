# `pd.core.groupby.Groupby.min`

`pandas.core.groupby.Groupby.min(numeric_only=False, min_count=-1)`


!!! note      
    * `min` is not supported on columns with nested array types
    * Categorical columns must be ordered.

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").min()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

       A      C
B
421  1.0  1.510
f31  2.0  2.421
```

