# `pd.Series.dt.nanosecond`

`pandas.Series.dt.nanosecond`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.nanosecond
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0       0
1     483
2     966
3     448
4     932
5     416
6     896
7     380
8     864
9     348
10    832
11    312
12    792
13    280
14    760
15    248
16    728
17    208
18    696
19    176
20    664
21    144
22    624
23    104
24    584
25     80
26    560
27     40
28    520
29      0
dtype: Int64
```

