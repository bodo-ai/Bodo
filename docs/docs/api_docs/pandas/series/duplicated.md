# `pd.Series.duplicated`

`pandas.Series.duplicated(keep='first')`

### Supported Arguments None

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...  return S.duplicated()
>
>>> S = pd.Series([1, 2, 1, np.nan, 3, 2, np.nan, 4])
0    False
1    False
2     True
3    False
4    False
5     True
6     True
7    False
dtype: bool
```
