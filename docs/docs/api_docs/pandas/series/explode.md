# `pd.Series.explode`

`pandas.Series.explode(ignore_index=False)`

### Supported Arguments None

!!! note
    Bodo's output type may differ from Pandas because Bodo must convert
    to a nullable type at compile time.


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.explode()
>>> S = pd.Series([np.arange(i) for i in range(10)])
>>> f(S)
0    <NA>
1       0
2       0
2       1
3       0
3       1
3       2
4       0
4       1
4       2
4       3
5       0
5       1
5       2
5       3
5       4
6       0
6       1
6       2
6       3
6       4
6       5
7       0
7       1
7       2
7       3
7       4
7       5
7       6
8       0
8       1
8       2
8       3
8       4
8       5
8       6
8       7
9       0
9       1
9       2
9       3
9       4
9       5
9       6
9       7
9       8
dtype: Int64
```

