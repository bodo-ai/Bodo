# `pd.TimedeltaIndex.nanoseconds`

`pandasTimedeltaIndex.nanoseconds`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.nanoseconds

>>> I = pd.TimedeltaIndex([pd.Timedelta(7, unit="nanos"))])
>>> f(I)
Int64Index([7], dtype='int64')
```

## PeriodIndex

`PeriodIndex` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.

## BinaryIndex

`BinaryIndex` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.

## MultiIndex
