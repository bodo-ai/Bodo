# `pd.DataFrame.duplicated`

`pandas.DataFrame.duplicated(subset=None, keep='first')`

### Supported Arguments : None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,1,3,4], "B": [1,1,3,3]})
...   return df.duplicated()
>>> f()
0    False
1     True
2    False
3    False
dtype: bool
```
