# `pd.Series.argsort`

`pandas.Series.argsort(axis=0, kind='quicksort', order=None)`

### Supported Arguments None

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.sort_values()
>>> S = pd.Series(np.arange(99, -1, -1), index=np.arange(100))
>>> f(S)
0     99
1     98
2     97
3     96
4     95
      ..
95     4
96     3
97     2
98     1
99     0
Length: 100, dtype: int64
```

