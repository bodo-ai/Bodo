# `pd.unique`

`pandas.unique(values)`

### Supported Arguments

| argument | datatypes |
|----------|----------------------------------------------|
| `values` | Series or 1-d array with Categorical dtypes |

### Example Usage

```py

>>> @bodo.jit
... def f(S):
...     return pd.unique(S)

>>> S = pd.Series([1, 2, 1, 3, 2, 1])
>>> f(S)
array([1, 2, 3])
```
