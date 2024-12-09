# `pd.DataFrame.fillna`


`pandas.DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)`

### Supported Arguments


- `value`: various scalars
    - Must be of the same type as the filled column
- `inplace`: Constant boolean
    - `inplace` is not supported alongside method
- `method`: One of `bfill`, `backfill`, `ffill` , or `pad`
    - **Must be constant at Compile Time**
    - `inplace` is not supported alongside method

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3,None], "B": [4, 5,None, None], "C": [6, None, None, None]})
...   return df.fillna(-1)
>>> f()
```

