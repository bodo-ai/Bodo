# `pd.DataFrame.values`

`pandas.DataFrame.values` (only for numeric dataframes)

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
...   return df.values
>>> f()
[[1.  3.1]
 [2.  4.2]
 [3.  5.3]]
```



