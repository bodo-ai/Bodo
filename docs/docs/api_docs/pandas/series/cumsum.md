# `pd.Series.cumsum`

`pandas.Series.cumsum(axis=None, skipna=True)`

### Supported Arguments None

!!! note
    - Series type must be numeric
    - Bodo does not accept any additional arguments for Numpy
    compatibility
    


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.cumsum()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0       0
1       1
2       3
3       6
4      10
     ...
95    283
96    288
97    294
98    294
99    295
Length: 100, dtype: int64
```

