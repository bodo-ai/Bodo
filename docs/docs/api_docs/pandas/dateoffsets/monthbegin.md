# `pd.tseries.offsets.MonthBegin`

`pandas.tseries.offsets.MonthBegin(n=1, normalize=False)`

### Supported Arguments

- `n`: integer
- `normalize`: boolean

### Example Usage

```py
>>> @bodo.jit
>>> def f(ts):
...     return ts + pd.tseries.offsets.MonthBegin(n=4, normalize=True)
>>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
>>> f(ts)

Timestamp('2021-02-01 00:00:00')
```
