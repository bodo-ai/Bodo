# `pd.Series.equals`

`pandas.Series.equals(other)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|---------------------------------------|
| `other` | Series |

!!! note
Series and `other` must contain scalar values in each row

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...     return S.equals(other)
>>> S = pd.Series(np.arange(100)) % 10
>>> other = pd.Series(np.arange(100)) % 5
>>> f(S, other)
False
```
