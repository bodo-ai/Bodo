# `pd.DataFrame.product`


`pandas.DataFrame.product(axis=None, skipna=None, level=None, numeric_only=None)`


### Supported Arguments


- `axis`: Integer (0 or 1)
    - **Must be constant at Compile Time**


### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.product(axis=1)
>>> f()
A      6
B    120
C    504
dtype: int64
```

