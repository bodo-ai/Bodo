# `pd.DataFrame.max`


`pandas.DataFrame.max(axis=None, skipna=None, level=None, numeric_only=None)`


### Supported Arguments

- `axis`: Integer (0 or 1)
    - **Must be constant at Compile Time**

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.max(axis=1)
>>> f()
0    7
1    8
2    9
```
!!! note
    Only supported for dataframes containing float, non-null int, and datetime64ns values.


