# `pd.Series.mad`

`pandas.Series.mad(axis=None, skipna=None, level=None)`

### Supported Arguments

| argument                    | datatypes                             |
|-----------------------------|---------------------------------------|
| `skipna`                    |   Boolean                             |

!!! note
    Series type must be numeric


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.mad()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
1.736
```

