# `pd.DataFrame.take`

`pandas.DataFrame.take(indices, axis=0, is_copy=None)`

### Supported Arguments

- indices: scalar Integer, Pandas Integer Array, Numpy Integer Array, Integer Series

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.take(pd.Series([-1,-2]))
>>> f()
   A  B  C
2  3  6  9
1  2  5  8
```
