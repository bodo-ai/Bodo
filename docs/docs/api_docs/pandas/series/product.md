# `pd.Series.product`

`pandas.Series.product(axis=None, skipna=None, level=None, numeric_only=None)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `skipna`                    |    Boolean                             |

!!! note
    - Series type must be numeric
    - Bodo does not accept any additional arguments to pass to the
    function


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.product()
>>> S = (pd.Series(np.arange(20)) % 3) + 1
>>> f(S)
93312
```

