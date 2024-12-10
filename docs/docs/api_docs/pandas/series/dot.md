# `pd.Series.dot`

`pandas.Series.dot(other)`

### Supported Arguments

| argument | datatypes |
|----------|--------------------------|
| `other` | Series with numeric data |

!!! note
`Series.dot` is only supported on Series of numeric data.

### Example Usage

```py
>>> @bodo.jit
... def f(S, other):
...   return S.dot(other)
>>> S = pd.Series(np.arange(1, 1001))
>>> other = pd.Series(reversed(np.arange(1, 1001)))
>>> f(S, other)
167167000
```

### Function application, GroupBy & Window
