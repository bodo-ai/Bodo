# `pd.tseries.offsets.MonthEnd`

`pandas.tseries.offsets.MonthEnd(n=1, normalize=False)`

### Supported Arguments

- `n`: integer
- `normalize`: boolean

### Example Usage

```py
>>> @bodo.jit
>>> def f(ts):
...     return ts + pd.tseries.offsets.MonthEnd(n=4, normalize=False)
>>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
>>> f(ts)

Timestamp('2021-01-31 22:00:00')
```
