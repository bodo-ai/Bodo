# `pd.Series.hasnans`

`pandas.Series.hasnans`

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.hasnans
>>> S = pd.Series(np.arange(1000))
>>> f(S)
False
```
