# `pd.Timestamp.day_name`


`pandasTimestamp.day_name(*args, **kwargs)`


### Supported Arguments: None

### Example Usage

```py

>>> @bodo.jit
... def f():
...   day_1 = pd.Timestamp(year=2021, month=12, day=9).day_name()
...   day_2 = pd.Timestamp(year=2021, month=12, day=10).day_name()
...   day_3 = pd.Timestamp(year=2021, month=12, day=11).day_name()
...   return (day_1, day_2, day_3)
>>> f()
('Thursday', 'Friday', 'Saturday')
```


