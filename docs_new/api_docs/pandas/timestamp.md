# Timestamp

Timestamp functionality is documented in [`pandas.Timestamp`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html){target=blank}.

#### `pd.Timestamp`


- <code><apihead>pandas.<apiname>Timestamp</apiname>(ts_input=<object object\>, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, *, fold=None)<apihead></code>

    ***Supported Arguments***

    - `ts_input`: string, integer, timestamp, datetimedate
    - `unit`: constant string
    - `year`: integer
    - `month`: integer
    - `day`: integer
    - `hour`: integer
    - `minute`: integer
    - `second`: integer
    - `microsecond`: integer
    - `nanosecond`: integer

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   return I.copy(name="new_name")
    ...   ts1 = pd.Timestamp('2021-12-09 09:57:44.114123')
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts3 = pd.Timestamp(100, unit="days")
    ...   ts4 = pd.Timestamp(datetime.date(2021, 12, 9), hour = 9, minute=57, second=44, microsecond=114123)
    ...   return (ts1, ts2, ts3, ts4)
    >>> f()
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 09:57:44.114123'), Timestamp('1970-04-11 00:00:00'), Timestamp('2021-12-09 09:57:44.114123'))
    ```
  
  
  
#### `pd.Timestamp.day`

- ++pandas.Timestamp.%%day%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day
    >>> f()
    9
    ```

#### `pd.Timestamp.hour`


- ++pandas.Timestamp.%%hour%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.hour
    >>> f()
    9
    ```
  
#### `pd.Timestamp.microsecond`


- ++pandas.Timestamp.%%microsecond%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.microsecond
    >>> f()
    114123
    ```
  
#### `pd.Timestamp.month`


- ++pandas.Timestamp.%%month%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.month
    >>> f()
    month
    ```

#### `pd.Timestamp.nanosecond`


- ++pandas.Timestamp.%%nanosecond%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(12, unit="ns")
    ...   return ts2.nanosecond
    >>> f()
    12
    ```
  
  
#### `pd.Timestamp.second`


- ++pandas.Timestamp.%%second%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.second
    >>> f()
    44
    ```
  
  
#### `pd.Timestamp.year`


- ++pandas.Timestamp.%%year%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.year
    >>> f()
    2021
    ```
  
  
#### `pd.Timestamp.dayofyear`


- ++pandas.Timestamp.%%dayofyear%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.dayofyear
    >>> f()
    343
    ```
  
  
#### `pd.Timestamp.day_of_year`


- ++pandas.Timestamp.%%day_of_year%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_year
    >>> f()
    343
    ```
  
  
#### `pd.Timestamp.dayofweek`


- ++pandas.Timestamp.%%dayofweek%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_year
    >>> f()
    343
    ```
  
  
#### `pd.Timestamp.day_of_week`


- ++pandas.Timestamp.%%day_of_week%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_week
    >>> f()
    3
    ```
  
  
#### `pd.Timestamp.days_in_month`


- ++pandas.Timestamp.%%days_in_month%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.days_in_month
    >>> f()
    31
    ```

#### `pd.Timestamp.daysinmonth`


- ++pandas.Timestamp.%%daysinmonth%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.daysinmonth
    >>> f()
    31
    ```
  
  
#### `pd.Timestamp.is_leap_year`


- ++pandas.Timestamp.%%is_leap_year%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2020, month=2,day=2)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return (ts1.is_leap_year, ts2.is_leap_year)
    >>> f()
    (True, False)
    ```
  
  
#### `pd.Timestamp.is_month_start`


- ++pandas.Timestamp.%%is_month_start%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=2)
    ...   return (ts1.is_month_start, ts2.is_month_start)
    >>> f()
    (True, False)
    ```
  
  
#### `pd.Timestamp.is_month_end`


- ++pandas.Timestamp.%%is_month_end%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=30)
    ...   return (ts1.is_month_end, ts2.is_month_end)
    >>> f()
    (True, False)
    ```
  
  
#### `pd.Timestamp.is_quarter_start`


- ++pandas.Timestamp.%%is_quarter_start%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
    ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
    ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
    >>> f()
    (False, True)
    ```
  
  
#### `pd.Timestamp.is_quarter_end`


- ++pandas.Timestamp.%%is_quarter_end%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
    ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
    ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
    >>> f()
    (True, False)
    ```
  
  
#### `pd.Timestamp.is_year_start`


- ++pandas.Timestamp.%%is_year_start%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
    ...   return (ts1.is_year_start, ts2.is_year_start)
    >>> f()
    (False, True)
    ```
  
  
#### `pd.Timestamp.is_year_end`


- ++pandas.Timestamp.%%is_year_end%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
    ...   return (ts1.is_year_end, ts2.is_year_end)
    >>> f()
    (True, False)
    ```
  
  
#### `pd.Timestamp.quarter`


- ++pandas.Timestamp.%%quarter%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=1)
    ...   return (ts1.quarter, ts2.quarter)
    >>> f()
    (4, 3)
    ```
  
  
#### `pd.Timestamp.week`


- ++pandas.Timestamp.%%week%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
    ...   return (ts1.week, ts2.week)
    >>> f()
    (35, 38)
    ```
  
  
#### `pd.Timestamp.weekofyear`


- ++pandas.Timestamp.%%weekofyear%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
    ...   return (ts1.weekofyear, ts2.weekofyear)
    >>> f()
    (35, 38)
    ```
  
  
#### `pd.Timestamp.value`


- ++pandas.Timestamp.%%value%%++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(12345, unit="ns").value
    >>> f()
    12345
    ```
  
  
#### `pd.Timestamp.ceil`


- ++pandas.Timestamp.%%ceil%%(freq, ambiguous='raise', nonexistent='raise')++

    ***Supported Arguments***

    - `freq`: string

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
    ...   return (ts1, ts2)
    >>> f()
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-10 00:00:00'))
    ```
  
  
#### `pd.Timestamp.date`


- ++pandas.Timestamp.%%date%%()++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).date()
    ...   return (ts1, ts2)
    >>> f()
    (Timestamp('2021-12-09 09:57:44.114123'), datetime.date(2021, 12, 9))
    ```
  

#### `pd.Timestamp.day_name`


- ++pandas.Timestamp.%%day_name%%(*args, **kwargs)++


    ***Supported Arguments***: None

    ***Example Usage***

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
  
  
#### `pd.Timestamp.floor`


- ++pandas.Timestamp.%%floor%%(freq, ambiguous='raise', nonexistent='raise')++

    ***Supported Arguments***

    - `freq`: string

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
    ...   return (ts1, ts2)
    >>> f()
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 00:00:00'))
    ```
  
  
#### `pd.Timestamp.isocalendar`


- ++pandas.Timestamp.%%isocalendar%%()++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
    ...   return (ts1, ts2)
    >>> f()
    (2021, 49, 4)
    ```
  
  
#### `pd.Timestamp.isoformat`


- ++pandas.Timestamp.%%isoformat%%()++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
    ...   return (ts1, ts2)
    >>> f()
    '2021-12-09T09:57:44'
    ```
  
  
#### `pd.Timestamp.month_name`


- ++pandas.Timestamp.%%month_name%%(locale=None)++

    ***Supported Arguments***: None

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9).month_name()
    >>> f()
    'December'
    ```
  
  
#### `pd.Timestamp.normalize`


- ++pandas.Timestamp.%%normalize%%()++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).normalize()
    ...   return (ts1, ts2)
    >>> f()
    Timestamp('2021-12-09 00:00:00')
    ```
  
  
#### `pd.Timestamp.round`


- ++pandas.Timestamp.%%round%%(freq, ambiguous='raise', nonexistent='raise')++

    ***Supported Arguments***

    - `freq`: string

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 12).round()
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 13).round()
    ...   return (ts1, ts2)
    >>> f()
    (Timestamp('2021-12-09 00:00:00'),Timestamp('2021-12-10 00:00:00'))
    ```
  

#### `pd.Timestamp.strftime`


- ++pandas.Timestamp.%%strftime%%(format)++

    ***Supported Arguments***

    - `format`: string

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9, hour = 12).strftime('%Y-%m-%d %X')
    >>> f()
    '2021-12-09 12:00:00'
    ```
  
  
#### `pd.Timestamp.toordinal`


- ++pandas.Timestamp.%%toordinal%%()++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9).toordinal()
    >>> f()
    738133
    ```
  
  
#### `pd.Timestamp.weekday`


- ++pandas.Timestamp.%%weekday%%()++

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=10)
    ...   return (ts1.weekday(), ts2.weekday())
    >>> f()
    (3, 4)
    ```
  
  
#### `pd.Timestamp.now`


- ++pandas.Timestamp.%%now%%(tz=None)++

    ***Supported Arguments***: None

    ***Example Usage***

    ```py

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp.now()
    >>> f()
    Timestamp('2021-12-10 10:54:06.457168')

    ```
  
