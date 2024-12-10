# `pd.DataFrame.insert`


`pandas.DataFrame.insert(loc, column, value, allow_duplicates=False)`

### Supported Arguments


- `loc`: constant integer
- `column`: constant string
- `value`: scalar, list/tuple, Pandas/Numpy array, Pandas index types, series
- `allow_duplicates`: constant boolean


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   df.insert(3, "D", [-1,-2,-3])
...   return df
>>> f()
  A  B  C  D
0  1  4  7 -1
1  2  5  8 -2
2  3  6  9 -3
```

