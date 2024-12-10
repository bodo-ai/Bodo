# `pd.Series.idxmin`

`pandas.Series.idxmin(axis=0, skipna=True)`

### Supported Arguments None

!!! note
    Bodo does not accept any additional arguments for Numpy
    compatibility


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.idxmin()
>>> S = pd.Series(np.arange(100))
>>> S[(S % 3 == 0)] = 100
>>> f(S)
1
```

