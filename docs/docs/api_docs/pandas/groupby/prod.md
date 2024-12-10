# `pd.core.groupby.Groupby.prod`

`pandas.core.groupby.Groupby.prod(numeric_only=NoDefault.no_default, min_count=0)`

!!! note
`prod` is not supported on columns with nested array types

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").prod()
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
421  7962624.0  5.417831e+27
f31       32.0  2.257108e+07
```
