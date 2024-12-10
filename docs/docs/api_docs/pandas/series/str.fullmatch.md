# `pd.Series.str.fullmatch`

`pandas.Series.str.fullmatch(pat, case=True, flags=0, na=np.nan)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|--------------------------------------|
| `pat` | String |
| `case` | Boolean |
| `flags` | Integer |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.fullmatch("ab.*")
>>> S = pd.Series(["abcdefg", "cab", "abc @123", "ABC", "Abcd"])
>>> f(S)
0     True
1     False
2     True
3     False
4     False
dtype: boolean
```
