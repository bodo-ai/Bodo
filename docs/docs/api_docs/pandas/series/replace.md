# `pd.Series.replace`

`pandas.Series.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')`

### Supported Arguments

| argument | datatypes | other requirements |
|-------------------|---------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `to_replace` | <ul><li> Scalar </li><li> List of Scalars </li><li> Dictionary mapping scalars of the same type </li></ul> | |
| `value` | - Scalar | If `to_replace` is not a scalar, value must be `None` |

### Example Usage

```py
>>> @bodo.jit
... def f(S, replace_dict):
...     return S.replace(replace_dict)
>>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
>>> f(S, {1: -2, -2: 5, 5: 27})
0    <NA>
1      -2
2    <NA>
3       5
4    <NA>
5      27
6    <NA>
dtype: Int64
```

### Reshaping, sorting
