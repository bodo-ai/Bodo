# `pd.TimedeltaIndex.seconds`

`pandasTimedeltaIndex.seconds`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.seconds

>>> I = pd.TimedeltaIndex([pd.Timedelta(-2, unit="S"))])
>>> f(I)
Int64Index([-2], dtype='int64')
```
