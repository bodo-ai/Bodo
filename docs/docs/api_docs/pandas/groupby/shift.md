# `pd.core.groupby.DataFrameGroupby.shift`

`pandas.core.groupby.DataFrameGroupby.shift(periods=1, freq=None, axis=0, fill_value=None)`

!!! note
`shift` is not supported on columns with nested array types

### Example Usage

```py

>>> @bodo.jit
... def f(df):
...     return df.groupby("B").shift()
>>> df = pd.DataFrame(
...      {
...          "A": [1, 2, 24, None] * 5,
...          "B": ["421", "f31"] * 10,
...          "C": [1.51, 2.421, 233232, 12.21] * 5
...      }
... )
>>> f(df)

       A           C
0    NaN         NaN
1    NaN         NaN
2    1.0       1.510
3    2.0       2.421
4   24.0  233232.000
5    NaN      12.210
6    1.0       1.510
7    2.0       2.421
8   24.0  233232.000
9    NaN      12.210
10   1.0       1.510
11   2.0       2.421
12  24.0  233232.000
13   NaN      12.210
14   1.0       1.510
15   2.0       2.421
16  24.0  233232.000
17   NaN      12.210
18   1.0       1.510
19   2.0       2.421
```
