# `pd.Series.dt.year`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html#pandas.Series.dt.year)

`pandas.Series.dt.year`

!!! note
Input must be a Series of `datetime64` data.

### Example Usage

```py
>>> @bodo.jit
... def f(S):
...     return S.dt.year
>>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
>>> f(S)
0     2022
1     2022
2     2022
3     2022
4     2022
5     2022
6     2022
7     2022
8     2022
9     2022
10    2023
11    2023
12    2023
13    2023
14    2023
15    2023
16    2023
17    2023
18    2023
19    2023
20    2024
21    2024
22    2024
23    2024
24    2024
25    2024
26    2024
27    2024
28    2024
29    2025
dtype: Int64
```
