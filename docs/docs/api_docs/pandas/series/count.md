# `pd.Series.count`

`pandas.Series.count(level=None)`

### Supported Arguments None

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.count()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
100
```
