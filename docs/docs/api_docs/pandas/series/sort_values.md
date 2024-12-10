# `pd.Series.sort_values`

`pandas.Series.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)`

### Supported Arguments

| argument      | datatypes                | other requirements                    |
|---------------|--------------------------|---------------------------------------|
| `ascending`   | Boolean                  |                                       |
| `na_position` | One of ("first", "last") | **Must be constant at  Compile Time** |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.sort_values()
>>> S = pd.Series(np.arange(99, -1, -1), index=np.arange(100))
>>> f(S)
99     0
98     1
97     2
96     3
95     4
      ..
4     95
3     96
2     97
1     98
0     99
Length: 100, dtype: int64
```

