# `pd.DateTimeIndex.is_leap_year`


`pandasDatetimeIndex.is_leap_year`

### Example Usage

```py

>>> @bodo.jit
... def f(I):
...   return I.is_leap_year

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
[Flase False False True True]
```


