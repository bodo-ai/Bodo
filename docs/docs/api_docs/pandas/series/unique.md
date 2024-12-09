# `pd.Series.unique`

`pandas.Series.unique()`


!!! note
    The output is assumed to be "small" relative to input and is
    replicated. Use `Series.drop_duplicates()` if the output should
    remain distributed.


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.unique()
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
[0 1 2 3 4 5 6]
```

