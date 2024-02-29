# `pd.DataFrame.to_numpy`

`pandas.DataFrame.to_numpy(dtype=None, copy=False, na_value=NoDefault.no_default)`

### Supported Arguments

- `copy`: boolean

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
...   return df.to_numpy()
>>> f()
[[1.  3.1]
 [2.  4.2]
 [3.  5.3]]
```

