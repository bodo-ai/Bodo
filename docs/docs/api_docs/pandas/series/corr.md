# `pd.Series.corr`

`pandas.Series.corr(other, method='pearson', min_periods=None)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `other` | Numeric Series or Array |

!!! note
Series type must be numeric

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...     return S.cov(other)
>>> S = pd.Series(np.arange(100)) % 7
>>> other = pd.Series(np.arange(100)) % 10
>>> f(S, other)
0.004326329627279103
```
