# `pd.Series.memory_usage`

`pandas.Series.memory_usage(index=True, deep=False)`

### Supported Arguments

| argument | datatypes | other requirements |
|----------|-----------|--------------------------------------|
| `index` | Boolean | **Must be constant at Compile Time** |

!!! note
This tracks the number of bytes used by Bodo which may differ from
the Pandas values.

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.memory_usage()
>>> S = pd.Series(np.arange(1000))
>>> f(S)
8024
```
