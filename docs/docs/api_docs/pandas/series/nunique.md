# `pd.Series.nunique`

`pandas.Series.nunique(dropna=True)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|---------------------------------------|
| `dropna` | Boolean |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.nunique()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
7
```
