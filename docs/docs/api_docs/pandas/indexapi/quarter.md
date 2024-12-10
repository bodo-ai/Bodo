# `pd.DateTimeIndex.quarter`

`pandasDatetimeIndex.quarter`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.quarter

>>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
>>> f(I)
Int64Index([4, 4, 4, 1, 1], dtype='int64')
```


Subtraction of `Timestamp` from `DatetimeIndex` and vice versa
is supported.

Comparison operators `==`, `!=`, `>=`, `>`, `<=`, `<` between
`DatetimeIndex` and a string of datetime
are supported.

## TimedeltaIndex

`TimedeltaIndex` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

