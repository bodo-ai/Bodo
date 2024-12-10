# `pd.core.groupby.DataFrameGroupby.idxmin`

`pandas.core.groupby.DataFrameGroupby.idxmin(axis=0, skipna=True)`

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").idxmin()
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
421  0  0
f31  1  1
```
