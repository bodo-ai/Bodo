# `pd.core.groupby.DataFrameGroupby.transform`

`pandas.core.groupby.DataFrameGroupby.transform(func, \*args, engine=None, engine_kwargs=None, **kwargs)`

### Supported Arguments

- `func`: Constant string, Python function from the builtins module that matches a supported operation
- Numpy functions cannot be provided.

!!! note
The supported builtin functions are `'count'`, `'first'`, `'last'`,
`'min'`, `'max'`, `'mean'`, `'median'`, `'nunique'`, `'prod'`,
`'std'`, `'sum'`, and `'var'`

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B", dropna=True).transform(max)
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

       A          C
0   24.0  233232.00
1    2.0      12.21
2   24.0  233232.00
3    2.0      12.21
4   24.0  233232.00
5    2.0      12.21
6   24.0  233232.00
7    2.0      12.21
8   24.0  233232.00
9    2.0      12.21
10  24.0  233232.00
11   2.0      12.21
12  24.0  233232.00
13   2.0      12.21
14  24.0  233232.00
15   2.0      12.21
16  24.0  233232.00
17   2.0      12.21
18  24.0  233232.00
19   2.0      12.21
```
