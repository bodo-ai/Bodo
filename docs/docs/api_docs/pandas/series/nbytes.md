# `pd.Series.nbytes`

`pandas.Series.nbytes`


!!! note
    This tracks the number of bytes used by Bodo which may differ from
    the Pandas values.


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.nbytes
>>> S = pd.Series(np.arange(1000))
>>> f(S)
8000
```

