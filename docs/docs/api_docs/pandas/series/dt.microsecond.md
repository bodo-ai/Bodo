# `pd.Series.dt.microsecond`

`pandas.Series.dt.microsecond`

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.microsecond
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0          0
1     931034
2     862068
3     793103
4     724137
5     655172
6     586206
7     517241
8     448275
9     379310
10    310344
11    241379
12    172413
13    103448
14     34482
15    965517
16    896551
17    827586
18    758620
19    689655
20    620689
21    551724
22    482758
23    413793
24    344827
25    275862
26    206896
27    137931
28     68965
29         0
dtype: Int64
```

