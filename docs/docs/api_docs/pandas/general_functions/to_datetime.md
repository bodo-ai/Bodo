# `pd.to_datetime`


`pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)`

### Supported Arguments

| argument                | datatypes                                                                                                                  | other requirements                                                                                                                                             |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arg`                   | Series, Array or scalar of integers  or strings                                                                            |                                                                                                                                                                |
| `errors`                | String and one of ('ignore', 'raise', 'coerce')                                                                            |                                                                                                                                                                |
| `dayfirst`              | Boolean                                                                                                                    |                                                                                                                                                                |
| `yearfirst`             | Boolean                                                                                                                    |                                                                                                                                                                |
| `utc`                   | Boolean                                                                                                                    |                                                                                                                                                                |
| `format`                | String matching Pandas [strftime/strptime](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) |                                                                                                                                                                |
| `exact`                 | Boolean                                                                                                                    |                                                                                                                                                                |
| `unit`                  | String                                                                                                                     | <ul><li> Must be a [valid Pandas timedelta unit](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) </li></ul> |
| `infer_datetime_format` | Boolean                                                                                                                    |                                                                                                                                                                |
| `origin`                | Scalar string or timestamp value                                                                                           |                                                                                                                                                                |
| `cache`                 | Boolean                                                                                                                    |                                                                                                                                                                |

!!! note

    * The function is not optimized.
    * Bodo doesn't support Timezone-Aware datetime values

### Example Usage

```py

>>> @bodo.jit
... def f(val):
...     return pd.to_datetime(val, format="%Y-%d-%m")

>>> val = "2016-01-06"
>>> f(val)

Timestamp('2016-06-01 00:00:00')
```

