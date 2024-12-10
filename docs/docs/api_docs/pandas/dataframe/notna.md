# `pd.DataFrame.notna`

`pandas.DataFrame.notna()`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,None,3]})
...   return df.notna()
>>> f()
       A
0   True
1  False
2   True
```
