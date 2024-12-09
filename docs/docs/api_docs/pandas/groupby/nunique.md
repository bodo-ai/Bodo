# `pd.core.groupby.DataFrameGroupby.nunique`

`pandas.core.groupby.DataFrameGroupby.nunique(dropna=True)`


### Supported Arguments
  
- `dropna`: boolean


!!! note
    `nunique` is not supported on columns with nested array types

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").nunique()
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
f31  1  2
```
  
