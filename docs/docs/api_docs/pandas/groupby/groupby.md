# `pd.DataFrame.groupby`

`pandas.DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)`

### Supported Arguments

- `by`: Column label or list of column labels
  - **Must be constant at Compile Time**
  - **This argument is required**
- `as_index`: Boolean
  - **Must be constant at Compile Time**
- `dropna`: Boolean
  - **Must be constant at Compile Time**

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B", dropna=True, as_index=False).count()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

     B   A   C
0  421  10  10
1  f31   5  10
```
