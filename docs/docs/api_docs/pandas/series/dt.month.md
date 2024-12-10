# `pd.Series.dt.month`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.month.html#pandas.Series.dt.month)

`pandas.Series.dt.month`

!!! note
Input must be a Series of `datetime64` data.

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.dt.month
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0      1
1      2
2      3
3      4
4      6
5      7
6      8
7      9
8     11
9     12
10     1
11     2
12     4
13     5
14     6
15     7
16     9
17    10
18    11
19    12
20     2
21     3
22     4
23     5
24     7
25     8
26     9
27    10
28    12
29     1
dtype: Int64
```
