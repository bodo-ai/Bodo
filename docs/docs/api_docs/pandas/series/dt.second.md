# `pd.Series.dt.second`

`pandas.Series.dt.second`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.second
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0      0
1     57
2     55
3     53
4     51
5     49
6     47
7     45
8     43
9     41
10    39
11    37
12    35
13    33
14    31
15    28
16    26
17    24
18    22
19    20
20    18
21    16
22    14
23    12
24    10
25     8
26     6
27     4
28     2
29     0
dtype: Int64
```

