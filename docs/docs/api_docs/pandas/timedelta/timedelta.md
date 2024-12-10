# `pd.Timedelta`


`pandas.Timedelta(value=<object object\>, unit="ns", days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)`


### Supported Arguments

- `value`: Integer (with constant string unit argument), String, Pandas Timedelta, datetime Timedelta
- `unit`: Constant String. Only has an effect when passing an integer `value`, see [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html) for allowed values.
- `days`: Integer
- `seconds`: Integer
- `microseconds`: Integer
- `milliseconds`: Integer
- `minutes`: Integer
- `hours`: Integer
- `weeks`: Integer

### Example Usage
```py
>>> @bodo.jit
... def f():
...   td1 = pd.Timedelta("10 Seconds")
...   td2 = pd.Timedelta(10, unit= "W")
...   td3 = pd.Timedelta(days= 10, hours=2, microseconds= 23)
...   return (td1, td2, td3)
>>> f()
(Timedelta('0 days 00:00:10'), Timedelta('70 days 00:00:00'), Timedelta('10 days 02:00:00.000023'))
```

