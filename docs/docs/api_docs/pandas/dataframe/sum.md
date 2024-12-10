# `pd.DataFrame.sum`


`pandas.DataFrame.sum(axis=None, skipna=None, level=None, numeric_only=None, min_count=0)`

### Supported Arguments


- `axis`: Integer (0 or 1)
    - **Must be constant at Compile Time**


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.sum(axis=1)
>>> f()
0    12
1    15
2    18
dtype: int64
```

