#`pd.to_timedelta`


`pandas.to_timedelta(arg, unit=None, errors='raise')`

### Supported Arguments

| argument           | datatypes          | other requirements                                                                                                                                           |
|--------------------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arg`              | Series, Array or   |                                                                                                                                                              |
|                    | scalar of integers |                                                                                                                                                              |
|                    | or strings         |                                                                                                                                                              |
| `unit`             | String             | <ul><li>Must be a valid Pandas[timedelta unit](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) </li></ul> |

!!! note
    Passing string data as `arg` is not optimized.

### Example Usage

```py

>>> @bodo.jit
... def f(S):
...     return pd.to_timedelta(S, unit="D")

>>> S = pd.Series([1.0, 2.2, np.nan, 4.2], [3, 1, 0, -2], name="AA")
>>> f(val)

3   1 days 00:00:00
1   2 days 04:48:00
0               NaT
-2   4 days 04:48:00
Name: AA, dtype: timedelta64[ns]
```

