# `pd.Series.dt.minute`

`pandas.Series.dt.minute`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.minute
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0      0
1     28
2     57
3     26
4     55
5     24
6     53
7     22
8     51
9     20
10    49
11    18
12    47
13    16
14    45
15    14
16    43
17    12
18    41
19    10
20    39
21     8
22    37
23     6
24    35
25     4
26    33
27     2
28    31
29     0
dtype: Int64
```

