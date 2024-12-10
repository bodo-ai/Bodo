# `pd.Series.cov`

`pandas.Series.cov(other, min_periods=None, ddof=1)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|-----------------------------------------|
| `other` | - Numeric Series or Array |
| `ddof` | - Integer |

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
0.025252525252525252
```
