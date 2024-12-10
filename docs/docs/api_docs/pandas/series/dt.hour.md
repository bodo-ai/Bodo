# `pd.Series.dt.hour`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.hour.html#pandas.Series.dt.hour)

`pandas.Series.dt.hour`

!!! note
Input must be a Series of `datetime64` data.

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.dt.hour
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0      0
1      2
2      4
3      7
4      9
5     12
6     14
7     17
8     19
9     22
10     0
11     3
12     5
13     8
14    10
15    13
16    15
17    18
18    20
19    23
20     1
21     4
22     6
23     9
24    11
25    14
26    16
27    19
28    21
29     0
dtype: Int64
```
