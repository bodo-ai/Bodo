# `pd.Series.dt.day_name`

`pandas.Series.dt.day_name(locale=None)`

### Supported Arguments None

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.day_name()
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
>>> f(S)
0      Saturday
1      Saturday
2      Saturday
3      Saturday
4        Sunday
5        Sunday
6        Sunday
7        Monday
8        Monday
9        Monday
10      Tuesday
11      Tuesday
12      Tuesday
13    Wednesday
14    Wednesday
15    Wednesday
16    Wednesday
17     Thursday
18     Thursday
19     Thursday
20       Friday
21       Friday
22       Friday
23     Saturday
24     Saturday
25     Saturday
26       Sunday
27       Sunday
28       Sunday
29       Monday
dtype: object
```

### String handling

