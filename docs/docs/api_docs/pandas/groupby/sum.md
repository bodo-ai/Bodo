# `pd.core.groupby.Groupby.sum`

`pandas.core.groupby.Groupby.sum(numeric_only=NoDefault.no_default, min_count=0)`


!!! note
    `sum` is not supported on columns with nested array types

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").sum()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

         A            C
B
421  125.0  1166167.550
f31   10.0       73.155
```

