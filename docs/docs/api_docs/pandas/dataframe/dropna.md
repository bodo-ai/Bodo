# `pd.DataFrame.dropna`

`pandas.DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)`

### Supported Arguments

- `how`: Constant String: either "all" or "any"
- `thresh`: Integer
- `subset`: Constant list/tuple of String column names, Constant list/tuple of Integer column names, Constant String column names, Constant Integer column names

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3,None], "B": [4, 5,None, None], "C": [6, None, None, None]})
...   df_1 = df.dropna(how="all", subset=["B", "C"])
...   df_2 = df.dropna(thresh=3)
...   formated_out = "\n".join([df_1.to_string(), df_2.to_string()])
...   return formated_out
>>> f()
   A  B     C
0  1  4     6
1  2  5  <NA>
   A  B  C
0  1  4  6
```
