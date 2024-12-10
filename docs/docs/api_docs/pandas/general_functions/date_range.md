# `pd.date_range`

`pandas.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs)`

### Supported Arguments

| argument  | datatypes           | other requirements                                                                                                                                        |
|-----------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start`   | String or Timestamp |                                                                                                                                                           |
| `end`     | String or Timestamp |                                                                                                                                                           |
| `periods` | Integer             |                                                                                                                                                           |
| `freq`    | String              | <ul><li> Must be a [valid Pandas frequency](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) </li></ul> |
| `name`    | String              |                                                                                                                                                           |

!!! note

    * Exactly three of `start`, `end`, `periods`, and `freq` must
      be provided.
    * Bodo **Does Not** support `kwargs`, even for compatibility.

### Example Usage

```py

>>> @bodo.jit
... def f():
...     return pd.date_range(start="2018-04-24", end="2018-04-27", periods=3)

>>> f()

DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',
              '2018-04-27 00:00:00'],
             dtype='datetime64[ns]', freq=None)
```

