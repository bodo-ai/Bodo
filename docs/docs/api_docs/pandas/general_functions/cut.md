# `pd.cut`

`pandas.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates="raise", ordered=True)`

### Supported Arguments

|argument | datatypes |
|-------------------------|--------------------------|
|`x` | Series or Array like |
|`bins` | Integer or Array like |
|`include_lowest` | Boolean |

### Example Usage

```py

 >>> @bodo.jit
 ... def f(S):
 ...   bins = 4
 ...   include_lowest = True
 ...   return pd.cut(S, bins, include_lowest=include_lowest)

 >>> S = pd.Series(
 ...    [-2, 1, 3, 4, 5, 11, 15, 20, 22],
 ...    ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"],
 ...    name="ABC",
 ... )
 >>> f(S)

a1    (-2.025, 4.0]
a2    (-2.025, 4.0]
a3    (-2.025, 4.0]
a4    (-2.025, 4.0]
a5      (4.0, 10.0]
a6     (10.0, 16.0]
a7     (10.0, 16.0]
a8     (16.0, 22.0]
a9     (16.0, 22.0]
Name: ABC, dtype: category
Categories (4, interval[float64, right]): [(-2.025, 4.0] < (4.0, 10.0] < (10.0, 16.0] < (16.0, 22.0]]
```
