# `pd.Series.dt.week`

`pandas.Series.dt.week`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.week
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0     52
1      6
2     11
3     17
4     22
5     27
6     33
7     38
8     44
9     49
10     3
11     8
12    14
13    19
14    24
15    30
16    35
17    41
18    46
19    52
20     5
21    11
22    16
23    21
24    27
25    32
26    38
27    43
28    49
29     2
dtype: Int64
```

