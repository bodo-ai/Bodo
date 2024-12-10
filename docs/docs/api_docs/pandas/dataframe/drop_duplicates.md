# `pd.DataFrame.drop_duplicates`

`pandas.DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)`

### Supported Arguments

- `subset`: Constant list/tuple of String column names, Constant list/tuple of Integer column names, Constant String column names, Constant Integer column names

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,1,3,4], "B": [1,1,3,3], "C": [7,8,9,10]})
...   return df.drop_duplicates(subset = ["A", "B"])
>>> f()
   A  B   C
0  1  1   7
2  3  3   9
3  4  3  10
```
