# `pd.DataFrame.astype`

`pandas.DataFrame.astype(dtype, copy=True, errors='raise')`

### Supported Arguments

- `dtype`: dict of string column names keys, and Strings/types values. String (string must be parsable by `np.dtype`), Valid type (see types), The following functions: float, int, bool, str

  - **Must be constant at Compile Time**

- `copy`: boolean

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
...   return df.astype({"A": float, "B": "datetime64[ns]"})
>>> f()
     A                             B
0  1.0 1970-01-01 00:00:00.000000003
1  2.0 1970-01-01 00:00:00.000000004
2  3.0 1970-01-01 00:00:00.000000005
```
