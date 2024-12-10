# `pd.Series.copy`

`pandas.Series.copy(deep=True)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|-----------------------------------------|
| `deep` | - Boolean |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.copy()
>>> S = pd.Series(np.arange(1000))
>>> f(S)
0        0
1        1
2        2
3        3
4        4
      ...
995    995
996    996
997    997
998    998
999    999
Length: 1000, dtype: int64
```
