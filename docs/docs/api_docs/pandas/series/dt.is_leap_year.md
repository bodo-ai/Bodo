# `pd.Series.dt.is_leap_year`

[Link to Pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_leap_year.html#pandas.Series.dt.is_leap_year)

`pandas.Series.dt.is_leap_year`

!!! note
	Input must be a Series of `datetime64` data.

### Example Usage

``` py
>>> @bodo.jit
... def f(S):
...     return S.dt.is_leap_year
>>> S = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')])
>>> f(S)
0      True
1     False
dtype: bool
```

