# `pd.DataFrame.index`

`pandas.DataFrame.index`

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3]}, index=["x", "y", "z"])
...   return df.index
>>> f()
Index(['x', 'y', 'z'], dtype='object')
```
