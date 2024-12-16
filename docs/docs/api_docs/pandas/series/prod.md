# `pd.Series.prod`

`pandas.Series.prod(axis=None, skipna=True, level=None, numeric_only=None, min_count=0)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `skipna`                    |    Boolean                             |
| `min_count`                 |    Integer                             |

!!! note
    - Series type must be numeric
    - Bodo does not accept any additional arguments to pass to the
    function


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.prod()
>>> S = (pd.Series(np.arange(20)) % 3) + 1
>>> f(S)
93312
```

