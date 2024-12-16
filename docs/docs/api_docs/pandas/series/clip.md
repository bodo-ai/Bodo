# `pd.Series.clip`

`pandas.Series.clip(axis=0, inplace=False, lower=None, upper=None)`

### Supported Arguments 

| argument    | datatypes                                  | other requirements                   |
|-------------|--------------------------------------------|--------------------------------------|
| `lower`     | Scalar or series matching the Series type  |                                      |
| `upper`     | Scalar or series matching the Series type  |                                      |

### Example Usage

``` py
>>> @bodo.jit
... def f(S, lower, upper):
...     return S.clip(lower, upper)
>>> S = pd.Series(pd.array([0, 1, 2, 3, 4, 5]), pd.array([1, 2, 2, 3, 1, 1]), pd.array([3, 3, 3, 3, 3, 4]))
>>> f(S)
0     1
1     2
2     2
3     3
4     3
5     4
dtype: Int64
```

