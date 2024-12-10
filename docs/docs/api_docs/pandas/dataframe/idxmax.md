# `pd.DataFrame.idxmax`

`pandas.DataFrame.idxmax(axis=0, skipna=True)`

### Supported Arguments : None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.idxmax()
>>> f()
A    2
B    2
C    2
dtype: int64
```
