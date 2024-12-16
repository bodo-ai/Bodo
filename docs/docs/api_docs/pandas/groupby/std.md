# `pd.core.groupby.Groupby.std`

`pandas.core.groupby.Groupby.std(ddof=1)`


!!! note
    `std` is only supported on numeric columns and is not supported on boolean column

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").std()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

             A              C
B
421  12.122064  122923.261366
f31   0.000000       5.159256
```

