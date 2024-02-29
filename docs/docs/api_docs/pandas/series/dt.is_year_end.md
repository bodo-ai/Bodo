# `pd.Series.dt.is_year_end`

`pandas.Series.dt.is_year_end`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.is_year_end
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
>>> f(S)
0     False
1     False
2     False
3     False
4     False
5     False
6     False
7     False
8     False
9     False
10    False
11    False
12    False
13    False
14    False
15    False
16    False
17    False
18    False
19    False
20    False
21    False
22    False
23    False
24    False
25    False
26    False
27    False
28    False
29     True
dtype: bool
```

