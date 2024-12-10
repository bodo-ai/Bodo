# `pd.DataFrame.reset_index`


`pandas.DataFrame.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')`

### Supported Arguments


- `level`: Integer
    - If specified, must drop all levels.
- `drop`: Constant boolean
- `inplace`: Constant boolean

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}, index = ["X", "Y", "Z"])
...   return df.reset_index()
>>> f()
  index  A  B  C
0     X  1  4  7
1     Y  2  5  8
2     Z  3  6  9
```

