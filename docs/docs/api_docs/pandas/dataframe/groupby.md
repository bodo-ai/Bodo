# `pd.DataFrame.groupby`


`pandas.DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)`

### Supported Arguments


- `by`: String column label,  List/Tuple of column labels
    - **Must be constant at Compile Time**
- `as_index`: boolean
    - **Must be constant at Compile Time**
- `dropna`: boolean
    - **Must be constant at Compile Time**


!!! note
    `sort=False` and `observed=True` are set by default. These are the only support values for sort and observed. For more information on using groupby, see [the groupby section][pd_groupby_section].


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,1,2,2], "B": [-2,-2,2,2]})
...   return df.groupby("A").sum()
>>> f()
   B
A
1 -4
2  4
```

