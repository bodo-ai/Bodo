# `pd.DataFrame.mean`


`pandas.DataFrame.mean(axis=None, skipna=None, level=None, numeric_only=None)`

### Supported Arguments


- `axis`: Integer (0 or 1)
    - **Must be constant at Compile Time**


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.mean(axis=1)
>>> f()
0    4.0
1    5.0
2    6.0
```

!!! note
  Only supported for dataframes containing float, non-null int, and datetime64ns values.

