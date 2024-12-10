# `pd.Series.to_frame`

`pandas.Series.to_frame(name=None)`

### Supported Arguments

| argument | datatypes | other requirements |
|-----------------|-----------------------|--------------------------------------|
| `name` | String | **Must be constant at Compile Time** |

!!! note
If `name` is not provided Series name must be a known constant

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.to_frame("my_column")
>>> S = pd.Series(np.arange(1000))
>>> f(S)
     my_column
0            0
1            1
2            2
3            3
4            4
..         ...
995        995
996        996
997        997
998        998
999        999
```

[1000 rows x 1 columns]
