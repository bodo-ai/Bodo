# `pd.core.groupby.Groupby.size`

`pandas.core.groupby.Groupby.size()`


### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").size()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

B
421    10
f31    10
dtype: int64
```
  
