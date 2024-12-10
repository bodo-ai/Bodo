# `pd.concat`

`pandas.concat(objs, axis=0, join="outer", join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True)`

### Supported Arguments

| argument | datatypes | other requirements |
|--------------------|-------------------------------------|-------------------------------------------------------------|
| `objs` | List or Tuple of DataFrames/Series | |
| `axis` | Integer with either 0 or 1 | <ul><li> **Must be constant at Compile Time** </li></ul> |
| `ignore_index` | Boolean | <ul><li> **Must be constant at Compile Time** </li></ul> |

!!! info "Important"
Bodo currently concatenates local data chunks for distributed datasets, which does not preserve global order of concatenated objects in output.

### Example Usage

```py

>>> @bodo.jit
... def f(df1, df2):
...     return pd.concat([df1, df2], axis=1)

>>> df1 = pd.DataFrame({"A": [3, 2, 1, -4, 7]})
>>> df2 = pd.DataFrame({"B": [3, 25, 1, -4, -24]})
>>> f(df1, df2)

A   B
0  3   3
1  2  25
2  1   1
3 -4  -4
4  7 -24
```
