# `pd.DataFrame.drop`

`pandas.DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')`

- Only dropping columns supported, either using `columns` argument or setting `axis=1` and using the `labels` argument
- `labels` and `columns` require constant string, or constant list/tuple of string values
- `inplace` supported with a constant boolean value
- All other arguments are unsupported

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   df.drop(columns = ["B", "C"], inplace=True)
...   return df
>>> f()
   A
0  1
1  2
2  3
```
