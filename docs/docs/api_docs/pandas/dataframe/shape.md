# `pd.DataFrame.shape`

`pandas.DataFrame.shape`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [3,4,5]})
...   return df.shape
>>> f()
(3, 2)
```

