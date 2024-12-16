# `pd.TimedeltaIndex.microseconds`

`pandasTimedeltaIndex.microseconds`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.microseconds

>>> I = pd.TimedeltaIndex([pd.Timedelta(11, unit="micros"))])
>>> f(I)
Int64Index([11], dtype='int64')
```

