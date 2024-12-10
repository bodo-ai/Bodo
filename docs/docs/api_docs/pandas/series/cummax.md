# `pd.Series.cummax`

`pandas.Series.cummax(axis=None, skipna=True)`

### Supported Arguments None

!!! note
    - Series type must be numeric
    - Bodo does not accept any additional arguments for Numpy
    compatibility


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.cummax()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0     0
1     1
2     2
3     3
4     4
     ..
95    6
96    6
97    6
98    6
99    6
Length: 100, dtype: int64
```

