# `pd.Series.loc`

`pandas.Series.loc`

- *getitem*:

  - `Series.loc` supports list/array of booleans
  - `Series.loc` supports integer with RangeIndex

- *setitem*:

  - `Series.loc` supports list/array of booleans

### Example Usage

```py
>>> @bodo.jit
... def f(S, idx):
...   return S.loc[idx]
>>> S = pd.Series(np.arange(1000))
>>> f(S, S < 10)
0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: int64
```

### Binary operator functions:
