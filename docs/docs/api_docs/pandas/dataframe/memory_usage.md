# `pd.DataFrame.memory_usage`

`pandas.DataFrame.memory_usage(index=True, deep=False)`

### Supported Arguments

- `index`: boolean

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": np.array([1,2,3], dtype=np.int64), "B": np.array([1,2,3], dtype=np.int32), "C": ["1", "2", "3456689"]})
...   return df.memory_usage()
>>> f()
Index    24
A        24
B        12
C        42
dtype: int64
```
