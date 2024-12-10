# `pd.Series.str.cat`

`pandas.Series.str.cat(others=None, sep=None, na_rep=None, join='left')`

### Supported Arguments

| argument | datatypes |
|-----------------|----------------------|
| `others` | DataFrame |
| `sep` | String |

### Example Usage

```py
>>> @bodo.jit
... def f(S, df):
...     return S.str.cat(df, ",")
>>> S = pd.Series(["s1", "s2", "s3", None, "s5"])
>>> df = pd.DataFrame({"A": ["a1", "a2", "a3", "a4", "a5"], "B": ["b1", "b2", None, "b4", "b5"]})
>>> f(S, df)
0    s1,a1,b1
1    s2,a2,b2
2         NaN
3         NaN
4    s5,a5,b5
dtype: object
```
