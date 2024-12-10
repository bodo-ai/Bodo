# `pd.Series.backfill`

`pandas.Series.backfill(axis=None, inplace=False, limit=None, downcast=None)`

### Supported Arguments None

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.backfill()
>>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
>>> f(S)
0       1
1       1
2      -2
3      -2
4       5
5       5
6    <NA>
dtype: Int64
```
