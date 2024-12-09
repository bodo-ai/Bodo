# `pd.Series.dt.quarter`

`pandas.Series.dt.quarter`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.quarter
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0     1
1     1
2     1
3     2
4     2
5     3
6     3
7     3
8     4
9     4
10    1
11    1
12    2
13    2
14    2
15    3
16    3
17    4
18    4
19    4
20    1
21    1
22    2
23    2
24    3
25    3
26    3
27    4
28    4
29    1
dtype: Int64
```

