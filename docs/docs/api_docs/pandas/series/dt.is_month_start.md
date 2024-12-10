# `pd.Series.dt.is_month_start`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_month_start.html#pandas.Series.dt.is_month_start)

`pandas.Series.dt.is_month_start`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.is_month_start
>>> SS = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
>>> f(S)
0      True
1     False
2     False
3     False
4      True
5     False
6     False
7     False
8     False
9     False
10    False
11    False
12    False
13    False
14    False
15    False
16    False
17    False
18    False
19    False
20    False
21    False
22    False
23    False
24    False
25     True
26    False
27    False
28    False
29    False
dtype: bool
```

