# `pd.Series.dt.day_of_week`

`pandas.Series.dt.day_of_week`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.day_of_week
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0     5
1     1
2     4
3     0
4     3
5     6
6     2
7     5
8     1
9     4
10    1
11    4
12    0
13    3
14    6
15    2
16    5
17    1
18    4
19    0
20    4
21    0
22    3
23    6
24    2
25    5
26    1
27    4
28    0
29    4
dtype: Int64
```

