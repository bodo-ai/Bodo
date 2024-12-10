# `pd.Series.sum`

`pandas.Series.sum(axis=None, skipna=None, level=None, numeric_only=None, min_count=0)`

### Supported Arguments

| argument                    | datatypes                             |
|-----------------------------|---------------------------------------|
| `skipna`                    |    Boolean                            |
| `min_count`                 |    Integer                            |

!!! note
    - Series type must be numeric
    - Bodo does not accept any additional arguments to pass to the
    function


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.sum()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
295
```

