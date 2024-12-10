# `pd.Series.reset_index`

`pandas.Series.reset_index(level=None, drop=False, name=None, inplace=False)`

### Supported Arguments

| argument                             | datatypes                                     | other requirements                                                                                                           |
|--------------------------------------|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `level`                              | <ul> <li> Integer</li><li> Boolean</li> </ul> |                                                                                                                              |
| **Must be constant at Compile Time** |                                               |                                                                                                                              |
| `drop`                               | Boolean                                       | <> <li> **Must be constant at Compile Time** </li> <li> If `False`, Index name must be known at compilation time </li> </ul> |

!!! note
    For MultiIndex case, only dropping all levels is supported.


### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.reset_index()
>>> S = pd.Series(np.arange(100), index=pd.RangeIndex(100, 200, 1, name="b"))
>>> f(S)
      b   0
0   100   0
1   101   1
2   102   2
3   103   3
4   104   4
..  ...  ..
95  195  95
96  196  96
97  197  97
98  198  98
99  199  99
>
[100 rows x 2 columns]
```

