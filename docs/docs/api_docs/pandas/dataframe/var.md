# `pd.DataFrame.var`


`pandas.DataFrame.var(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)`


### Supported Arguments

- `axis`: Integer (0 or 1)
    - **Must be constant at Compile Time**


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.var(axis=1)
>>> f()
0    9.0
1    9.0
2    9.0
dtype: float64
```

