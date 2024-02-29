# `pd.core.groupby.DataFrameGroupby.idxmax`


`pandas.core.groupby.DataFrameGroupby.idxmax(axis=0, skipna=True)`

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").idxmax()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

     A  C
B
421  2  2
f31  1  3
```

