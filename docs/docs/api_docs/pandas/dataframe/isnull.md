# `pd.DataFrame.isnull`

`pandas.DataFrame.isnull()`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,None,3]})
...   return df.isnull()
>>> f()
       A
0  False
1   True
2  False
```

