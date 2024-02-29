# `pd.Timestamp`


- pandas.Timestamp(ts_input=<object object\>, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, *, fold=None)

### Supported Arguments

- `ts_input`: string, integer, timestamp, datetimedate
- `unit`: constant string
- `year`: integer
- `month`: integer
- `day`: integer
- `hour`: integer
- `minute`: integer
- `second`: integer
- `microsecond`: integer
- `nanosecond`: integer

### Example Usage

```py

>>> @bodo.jit
... def f():
...   return I.copy(name="new_name")
...   ts1 = pd.Timestamp('2021-12-09 09:57:44.114123')
...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
...   ts3 = pd.Timestamp(100, unit="days")
...   ts4 = pd.Timestamp(datetime.date(2021, 12, 9), hour = 9, minute=57, second=44, microsecond=114123)
...   return (ts1, ts2, ts3, ts4)
>>> f()
(Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 09:57:44.114123'), Timestamp('1970-04-11 00:00:00'), Timestamp('2021-12-09 09:57:44.114123'))
```



