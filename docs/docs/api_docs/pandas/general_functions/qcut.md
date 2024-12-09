
# `pd.qcut`


`pandas.qcut(x, q, labels=None, retbins=False, precision=3, duplicates="raise")`

### Supported Arguments

|argument                   |  datatypes                        |
|---------------------------|-----------------------------------|
|`x`                        |  Series or Array like             |
|`q`                        |  Integer or Array like of floats  |

### Example Usage

```py

 >>> @bodo.jit
 ... def f(S):
 ...   q = 4
 ...   return pd.qcut(S, q)

 >>> S = pd.Series(
 ...      [-2, 1, 3, 4, 5, 11, 15, 20, 22],
 ...      ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"],
 ...      name="ABC",
 ... )
 >>> f(S)

 a1    (-2.001, 3.0]
 a2    (-2.001, 3.0]
 a3    (-2.001, 3.0]
 a4       (3.0, 5.0]
 a5       (3.0, 5.0]
 a6      (5.0, 15.0]
 a7      (5.0, 15.0]
 a8     (15.0, 22.0]
 a9     (15.0, 22.0]
 Name: ABC, dtype: category
 Categories (4, interval[float64, right]): [(-2.001, 3.0] < (3.0, 5.0] < (5.0, 15.0] < (15.0, 22.0]]
```
