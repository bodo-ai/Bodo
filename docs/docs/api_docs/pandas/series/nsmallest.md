# `pd.Series.nsmallest`

`pandas.Series.nsmallest(n=5, keep='first')`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `n`                         |    Integer                             |

!!! note
    Series type must be numeric


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.nsmallest(20)
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
63    0
7     0
56    0
98    0
77    0
91    0
49    0
42    0
35    0
84    0
28    0
21    0
70    0
0     0
14    0
43    1
1     1
57    1
15    1
36    1
dtype: int64
```

