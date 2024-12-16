# `pd.core.groupby.Groupby.var`

`pandas.core.groupby.Groupby.var(ddof=1)`


!!! note
    `var` is only supported on numeric columns and is not supported on boolean column

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").var()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

              A             C
B
421  146.944444  1.511013e+10
f31    0.000000  2.661792e+01
```

