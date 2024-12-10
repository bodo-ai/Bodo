# `pd.DateTimeIndex.is_year_end`

`pandasDatetimeIndex.is_year_end`

### Example Usage

```
>>> @bodo.jit
... def f(I):
...   return I.is_year_end

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([1, 1, 1, 0, 0], dtype='int64')
```

