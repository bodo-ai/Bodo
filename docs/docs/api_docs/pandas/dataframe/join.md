# `pd.DataFrame.join`

`pandas.DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)`

### Supported Arguments

- `other`: DataFrame
- `on`: constant string column name, constant list/tuple of column names

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
...   return df.join(on = "A", other=pd.DataFrame({"C": [-1,-2,-3], "D": [4,5,6]}))
>>> f()
   A  B     C     D
0  1  4    -2     5
1  1  5    -2     5
2  3  6  <NA>  <NA>

```

!!! note
Joined dataframes cannot have common columns. The output dataframe is not sorted by default for better parallel performance
