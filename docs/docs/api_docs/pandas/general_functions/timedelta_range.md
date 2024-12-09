# `pd.timedelta_range`


`pandas.timedelta_range(start=None, end=None, periods=None, freq=None, name=None, closed=None)`


### Supported Arguments

| argument  | datatypes                             | other requirements                                                                                                                                              |
|-----------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start`   | String or  Timedelta                  |                                                                                                                                                                 |
| `end`     | String or  Timedelta                  |                                                                                                                                                                 |
| `periods` | Integer                               |                                                                                                                                                                 |
| `freq`    | String                                | <ul><li>   Must be a [valid    Pandas  frequency](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) </li></ul> |
| `name`    | String                                |                                                                                                                                                                 |
| `closed`  | String and one of   ('left', 'right') |                                                                                                                                                                 |


!!! note
    * Exactly three of `start`, `end`, `periods`, and `freq` must be provided.
    * This function is not parallelized yet.

### Example Usage

```py

>>> @bodo.jit
... def f():
...     return pd.timedelta_range(start="1 day", end="11 days 1 hour", periods=3)

>>> f()

TimedeltaIndex(['1 days 00:00:00', '6 days 00:30:00', '11 days 01:00:00'], dtype='timedelta64[ns]', freq=None)

```
