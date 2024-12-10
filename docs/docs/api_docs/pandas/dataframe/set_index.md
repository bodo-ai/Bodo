# `pd.DataFrame.set_index`

`pandas.DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)`

### Supported Arguments

- keys: must be a constant string

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}, index = ["X", "Y", "Z"])
...   return df.set_index("C")
>>> f()
   A  B
C
7  1  4
8  2  5
9  3  6
```
