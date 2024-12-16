# `pd.Series.to_dict`

`pandas.Series.to_dict(into=<class 'dict'>)`

### Supported Arguments None

!!! note
    -   This method is not parallelized since dictionaries are not
        parallelized.
    - This method returns a typedDict, which maintains typing
    information if passing the dictionary between JIT code and regular
    Python. This can be converted to a regular Python dictionary by
    using the `dict` constructor.


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.to_dict()
>>> S = pd.Series(np.arange(10))
>>> dict(f(S))
{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
```

