# `pd.core.groupby.Groupby.head`

`pandas.core.groupby.Groupby.head(n=5)`


### Supported Arguments

- `n`: Non-negative integer
    - **Must be constant at Compile Time**


### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").head()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

      A    B           C
0   1.0  421       1.510
1   2.0  f31       2.421
2  24.0  421  233232.000
3   NaN  f31      12.210
4   1.0  421       1.510
5   2.0  f31       2.421
6  24.0  421  233232.000
7   NaN  f31      12.210
8   1.0  421       1.510
9   2.0  f31       2.421
```
  
