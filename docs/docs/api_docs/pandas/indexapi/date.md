# `pd.DateTimeIndex.date`


`pandasDatetimeIndex.date`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.date

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
[datetime.date(2019, 12, 31) datetime.date(2019, 12, 31) datetime.date(2019, 12, 31) datetime.date(2020, 1, 1) datetime.date(2020, 1, 1)]
```


