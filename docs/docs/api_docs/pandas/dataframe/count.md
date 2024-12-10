# `pd.DataFrame.count`

`pandas.DataFrame.count(axis=0, level=None, numeric_only=False)`

### Supported Arguments : None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1, None, 3], "B": [None, 2, None]})
...   return df.count()
>>> f()
A    2
B    1
```
