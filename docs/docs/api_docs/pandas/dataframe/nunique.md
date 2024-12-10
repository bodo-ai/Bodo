# `pd.DataFrame.nunique`

`pandas.DataFrame.nunique(axis=0, dropna=True)`

### Supported Arguments

- `dropna`: boolean

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [1,1,1], "C": [4, None, 6]})
...   return df.nunique()
>>> f()
A    3
B    1
C    2
```
