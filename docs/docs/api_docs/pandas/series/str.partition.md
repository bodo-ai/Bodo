# `pd.Series.str.partition`

`pandas.Series.str.partition(sep=' ', expand=True)`

### Supported Arguments

| argument                    | datatypes                              |
|-----------------------------|----------------------------------------|
| `sep`                       |    String                              |
| `expand`                    |    Boolean                             |

!!! note
    Bodo currently only supports expand=True.

``` py
>>> @bodo.jit
... def f(S):
...     return S.str.partition()
>>> S = pd.Series(["alphabet soup is delicious", "hello     world", "goodbye"])
>>> f(S)
          0  1                  2
0  alphabet     soup is delicious
1     hello                 world
2   goodbye                      
```

