# `pd.Series.sort_index`

`pandas.Series.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)`

### Supported Arguments

| argument      | datatypes                | other requirements                    |
|---------------|--------------------------|---------------------------------------|
| `ascending`   | Boolean                  |                                       |
| `na_position` | One of ("first", "last") | **Must be constant at  Compile Time** |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.sort_index()
>>> S = pd.Series(np.arange(100), index=np.arange(99, -1, -1))
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

