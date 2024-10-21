# `pd.Series.dt.month_name`

`pandas.Series.dt.month_name(locale=None)`

### Argument Restrictions:
 * `locale`: only supports default value `None`.

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.month_name()
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0       January
1      February
2         March
3         April
4          June
5          July
6        August
7     September
8      November
9      December
10      January
11     February
12        April
13          May
14         June
15         July
16    September
17      October
18     November
19     December
20     February
21        March
22        April
23          May
24         July
25       August
26    September
27      October
28     December
29      January
dtype: object
```

