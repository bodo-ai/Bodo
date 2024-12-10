# `pd.tseries.offsets.Week`

`pandas.tseries.offsets.Week(n=1, normalize=False, weekday=None)`

### Supported Arguments

- `n`: integer
- `normalize`: boolean
- `weekday`: integer

### Example Usage

```py
>>> @bodo.jit
>>> def f(ts):
...     return ts + pd.tseries.offsets.Week(n=4, normalize=True, weekday=5)
>>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
>>> f(ts)

Timestamp('2020-11-21 00:00:00')
```
