# `pd.Series.between`

`pandas.Series.between(left, right, inclusive='both')`

### Supported Arguments

| argument    | datatypes                        | other requirements                   |
|-------------|----------------------------------|--------------------------------------|
| `left`      | Scalar matching the Series type  |                                      |
| `right`     | Scalar matching  the Series type |                                      |
| `inclusive` | One of ("both", "neither")       | **Must be constant at Compile Time** |

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.between(3, 5, "both")
>>> S = pd.Series(np.arange(100)) % 7
>>> f(S)
0     False
1     False
2     False
3      True
4      True
      ...
95     True
96     True
97    False
98    False
99    False
Length: 100, dtype: bool
```

