# `pd.Series.value_counts`

`pandas.Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)`

### Supported Arguments

| argument | datatypes | other requirements |
|-------------|----------------------------------------------------------------|--------------------------------------|
| `normalize` | Boolean | **Must be constant at Compile Time** |
| `sort` | Boolean | **Must be constant at Compile Time** |
| `ascending` | Boolean | |
| `bins` | <ul><li> Integer </li></li> Array-like of integers </li></ul> | |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.value_counts()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0    15
1    15
2    14
3    14
4    14
5    14
6    14
dtype: int64
```

### Reindexing / Selection / Label manipulation
