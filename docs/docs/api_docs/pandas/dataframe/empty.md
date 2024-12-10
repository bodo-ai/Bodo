# `pd.DataFrame.empty`

`pandas.DataFrame.empty`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df1 = pd.DataFrame({"A": [1,2,3]})
...   df2 = pd.DataFrame()
...   return df1.empty, df2.empty
>>> f()
(False, True)
```
