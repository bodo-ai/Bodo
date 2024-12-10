# `pd.core.groupby.Groupby.max`

`pandas.core.groupby.Groupby.max(numeric_only=False, min_count=-1)`

!!! note
\* `max` is not supported on columns with nested array types.
\* Categorical columns must be ordered.

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").max()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

        A          C
B
421  24.0  233232.00
f31   2.0      12.21
```
