# `pd.Series.iloc`

`pandas.Series.iloc`

- *getitem*:

    -  `Series.iloc` supports single integer indexing (returns a
        scalar) `S.iloc[0]`
    -  `Series.iloc` supports list/array/series of integers/bool
        (returns a Series) `S.iloc[[0,1,2]]`
    -  `Series.iloc` supports integer slice (returns a Series)
        `S.iloc[[0:2]]`

- *setitem*:

    -   Supports the same cases as getitem but the array type must be
        mutable (i.e. numeric array)

### Example Usage

``` py
>>> @bodo.jit
... def f(S, idx):
...   return S.iloc[idx]
>>> S = pd.Series(np.arange(1000))
>>> f(S, [1, 4, 29])
1      1
4      4
29    29
dtype: int64
```

