# `pd.DataFrame.apply`


`pandas.DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), _bodo_inline=False, \**kwargs)`

### Supported Arguments


- `func`: function (e.g. lambda) (axis must = 1), jit function (axis must = 1), String which refers to a supported DataFrame method
    - **Must be constant at Compile Time**
- `axis`: Integer (0, 1), String (only if the method takes axis as an argument )
    - **Must be constant at Compile Time**
- `_bodo_inline`: boolean
    - **Must be constant at Compile Time**

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
...   return df.apply(lambda x: x["A"] * (x["B"] + x["C"]))
>>> f()
0    11
1    26
2    45
dtype: int64
```


!!! note

    Supports extra `_bodo_inline` boolean argument to manually control bodo's inlining behavior.
    Inlining user-defined functions (UDFs) can potentially improve performance at the expense of
    extra compilation time. Bodo uses heuristics to make a decision automatically if `_bodo_inline` is not provided.

