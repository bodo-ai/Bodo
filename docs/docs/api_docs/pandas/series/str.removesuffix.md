# `pd.Series.str.removesuffix`

`pandas.Series.str.removesuffix(suffix)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `suffix` | String |

```py
>>> @bodo.jit
... def f(S):
...     return S.str.removesuffix("b")
>>> S = pd.Series(["a", "ab", "abc", " abcd", "a bcd", "abcd", "xab"])
>>> f(S)
0        a
1        a
2      abc
3     abcd
4    a bcd
5     abcd
6       xa
dtype: string
```
