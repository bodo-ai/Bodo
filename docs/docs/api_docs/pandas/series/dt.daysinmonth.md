# `pd.Series.dt.daysinmonth`

`pandas.Series.dt.daysinmonth`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.daysinmonth
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
>>> f(S)
0     31
1     28
2     31
3     30
4     30
5     31
6     31
7     30
8     31
9     31
10    31
11    28
12    31
13    31
14    30
15    31
16    31
17    31
18    30
19    31
20    31
21    31
22    30
23    31
24    30
25    31
26    30
27    31
28    30
29    31
dtype: Int64
```

