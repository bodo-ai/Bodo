# `pd.Series.nlargest`

`pandas.Series.nlargest(n=5, keep='first')`

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
...     return S.nlargest(20)
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
20    6
27    6
41    6
34    6
55    6
13    6
83    6
90    6
6     6
69    6
48    6
76    6
62    6
97    6
19    5
5     5
26    5
61    5
12    5
68    5
dtype: int64
```

