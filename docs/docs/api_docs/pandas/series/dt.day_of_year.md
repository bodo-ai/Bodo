# `pd.Series.dt.day_of_year`

`pandas.Series.dt.day_of_year`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.day_of_year
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0       1
1      39
2      77
3     115
4     153
5     191
6     229
7     267
8     305
9     343
10     17
11     55
12     93
13    131
14    169
15    207
16    245
17    283
18    321
19    359
20     33
21     71
22    109
23    147
24    185
25    223
26    261
27    299
28    337
29     10
dtype: Int64
```

