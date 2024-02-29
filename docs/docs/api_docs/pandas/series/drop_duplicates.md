# `pd.Series.drop_duplicates`

`pandas.Series.drop_duplicates(keep='first', inplace=False)`

### Supported Arguments None

### Example Usage
>
``` py
>>> @bodo.jit
... def f(S):
...     return S.drop_duplicates()
>>> S = pd.Series(np.arange(100)) % 10
>>> f(S)
0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: int64
```

