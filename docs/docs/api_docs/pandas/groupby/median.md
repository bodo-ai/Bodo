# `pd.core.groupby.Groupby.median`

`pandas.core.groupby.Groupby.median(numeric_only=NoDefault.no_default)`


!!! note
    `median` is only supported on numeric columns and is not supported on boolean column


### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").median()
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
421  12.5  116616.7550
f31   2.0       7.3155
```
  
