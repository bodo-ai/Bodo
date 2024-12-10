# `pd.DataFrame.rename`

`pandas.DataFrame.rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore')`

### Supported Arguments

- `mapper`: must be constant dictionary.
  - Can only be used alongside axis=1
- `columns`: must be constant dictionary
- `axis`: Integer
  - Can only be used alongside mapper argument
- `copy`: boolean
- `inplace`: must be constant boolean

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.rename(columns={"A": "X", "B":"Y", "C":"Z"})
>>> f()
   X  Y  Z
0  1  4  7
1  2  5  8
2  3  6  9
```
