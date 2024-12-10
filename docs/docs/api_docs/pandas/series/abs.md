# `pd.Series.abs`

`pandas.Series.abs()`

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.abs()
>>> S = (pd.Series(np.arange(100)) % 7) - 2
>>> f(S)
0     2
1     1
2     0
3     1
4     2
     ..
95    2
96    3
97    4
98    2
99    1
Length: 100, dtype: int64
```
