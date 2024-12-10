# `pd.Series.tail`

`pandas.Series.tail(n=5)`

### Supported Arguments

| argument | datatypes |
|-----------------------------|----------------------------------------|
| `n` | Integer |

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.tail(10)
>>> S = pd.Series(np.arange(100))
>>> f(S)
90    90
91    91
92    92
93    93
94    94
95    95
96    96
97    97
98    98
99    99
dtype: int64
```
