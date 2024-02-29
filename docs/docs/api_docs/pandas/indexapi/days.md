# `pd.TimedeltaIndex.days`

`pandasTimedeltaIndex.days`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.days

>>> I = pd.TimedeltaIndex([pd.Timedelta(3, unit="D"))])
>>> f(I)
Int64Index([3], dtype='int64')
```

