# `pd.Series.dt.day`

`pandas.Series.dt.day`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.day
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0      1
1      8
2     18
3     25
4      2
5     10
6     17
7     24
8      1
9      9
10    17
11    24
12     3
13    11
14    18
15    26
16     2
17    10
18    17
19    25
20     2
21    11
22    18
23    26
24     3
25    10
26    17
27    25
28     2
29    10
dtype: Int64
```

