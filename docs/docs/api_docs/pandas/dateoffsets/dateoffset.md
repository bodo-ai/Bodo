# `pd.tseries.offsets.DateOffset`

`pandas.tseries.offsets.DateOffset(n=1, normalize=False, years=None, months=None, weeks=None, days=None, hours=None, minutes=None, seconds=None, microseconds=None, nanoseconds=None, year=None, month=None, day=None, weekday=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None)`

### Supported Arguments

- `n`: integer
- `normalize`: boolean
- `years`: integer
- `months`: integer
- `weeks`: integer
- `days`: integer
- `hours`: integer
- `minutes`: integer
- `seconds`: integer
- `microseconds`: integer
- `nanoseconds`: integer
- `year`: integer
- `month`: integer
- `weekday`: integer
- `day`: integer
- `hour`: integer
- `minute`: integer
- `second`: integer
- `microsecond`: integer
- `nanosecond`: integer

### Example Usage

```py
>>> @bodo.jit
>>> def f(ts):
...     return ts + pd.tseries.offsets.DateOffset(n=4, normalize=True, weeks=11, hour=2)
>>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
>>> f(ts)

Timestamp('2021-09-03 02:00:00')
```
