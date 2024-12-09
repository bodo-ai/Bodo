# `pd.DataFrame.melt`


`pandas.DataFrame.melt(id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None)`


### Supported Arguments

- `id_vars`: Constant Column label or list of labels
- `value_vars`: Constant Column label or list of labels

### Example Usage

```py
>>> @bodo.jit
... def f(df, id_vars, value_vars):
...   return df.melt(id_vars, value_vars)
>>> df = pd.DataFrame({"A": ["a", "b", "c"], 'B': [1, 3, 5], 'C': [2, 4, 6]})
>>> f(df, ["A"], ["B", "C"])
    A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
3  a        C      2
4  b        C      4
5  c        C      6
```

!!! note
    To offer increased performance, row ordering and corresponding Index value may not match Pandas when run on multiple cores.



