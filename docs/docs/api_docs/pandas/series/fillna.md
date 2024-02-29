# `pd.Series.fillna`

`pandas.Series.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)`

### Supported Arguments

| argument  | datatypes                                        | other requirements                   |
|-----------|--------------------------------------------------|--------------------------------------|
| `value`   | Scalar                                           |                                      |
| `method`  | One of ("bfill", "backfill", "ffill", and "pad") | **Must be constant at Compile Time** |
| `inplace` | Boolean                                          | **Must be constant at Compile Time** |

-   If `value` is provided then `method` must be `None` and
    vice-versa
-   If `method` is provided then `inplace` must be `False`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.fillna(-1)
>>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
>>> f(S)
0    -1
1     1
2    -1
3    -2
4    -1
5     5
6    -1
dtype: Int64
```

