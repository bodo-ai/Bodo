# `pd.DataFrame.append`

`pandas.DataFrame.append(other, ignore_index=False, verify_integrity=False, sort=False)`


### Supported Arguments

- `other`: DataFrame, list/tuple of DataFrame
- `ignore_index`: constant boolean

### Example Usage

```py
>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
...   return df.append(pd.DataFrame({"A": [-1,-2,-3], "C": [4,5,6]}))
>>> f()
   A    B    C
0  1  4.0  NaN
1  2  5.0  NaN
2  3  6.0  NaN
0 -1  NaN  4.0
1 -2  NaN  5.0
2 -3  NaN  6.0
```

