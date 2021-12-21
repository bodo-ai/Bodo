

Timestamp
~~~~~~~~~

Timestamp functionality is documented in `pandas.Timestamp <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html>`_.

* :class:`pandas.Timestamp` ``(ts_input=<object object>, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, *, fold=None)``

`Supported arguments`:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - argument
     - datatypes
   * - ``ts_input``
     - - string
       - integer
       - timestamp
       - datetimedate
   * - ``unit``
     - constant string
   * - ``year``
     - integer
   * - ``month``
     - integer
   * - ``day``
     - integer
   * - ``hour``
     - integer
   * - ``minute``
     - integer
   * - ``second``
     - integer
   * - ``microsecond``
     - integer
   * - ``nanosecond``
     - integer

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   return I.copy(name="new_name")
    ...   ts1 = pd.Timestamp('2021-12-09 09:57:44.114123')
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts3 = pd.Timestamp(100, unit="days")
    ...   ts4 = pd.Timestamp(datetime.date(2021, 12, 9), hour = 9, minute=57, second=44, microsecond=114123)
    ...   return (ts1, ts2, ts3, ts4)
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 09:57:44.114123'), Timestamp('1970-04-11 00:00:00'), Timestamp('2021-12-09 09:57:44.114123'))


* :attr:`pandas.Timestamp.day`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day
    9

* :attr:`pandas.Timestamp.hour`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.hour
    9

* :attr:`pandas.Timestamp.microsecond`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.microsecond
    114123

* :attr:`pandas.Timestamp.month`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.month
    month

* :attr:`pandas.Timestamp.nanosecond`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(12, unit="ns")
    ...   return ts2.nanosecond
    12

* :attr:`pandas.Timestamp.second`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.second
    44

* :attr:`pandas.Timestamp.year`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.year
    2021

* :attr:`pandas.Timestamp.dayofyear`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.dayofyear
    343
    
* :attr:`pandas.Timestamp.day_of_year`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_year
    343
    
* :attr:`pandas.Timestamp.dayofweek`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_year
    343
    
* :attr:`pandas.Timestamp.day_of_week`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_week
    3
    
* :attr:`pandas.Timestamp.days_in_month`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.days_in_month
    31
    
* :attr:`pandas.Timestamp.daysinmonth`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.daysinmonth
    31

* :attr:`pandas.Timestamp.is_leap_year`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2020, month=2,day=2)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return (ts1.is_leap_year, ts2.is_leap_year)
    (True, False)
    
* :attr:`pandas.Timestamp.is_month_start`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=2)
    ...   return (ts1.is_moth_start, ts2.is_moth_start)
    (True, False)

* :attr:`pandas.Timestamp.is_month_end`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=30)
    ...   return (ts1.is_moth_end, ts2.is_moth_end)
    (True, False)

* :attr:`pandas.Timestamp.is_quarter_start`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
    ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
    ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
    (False, True)

* :attr:`pandas.Timestamp.is_quarter_end`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
    ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
    ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
    (True, False)

* :attr:`pandas.Timestamp.is_year_start`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
    ...   return (ts1.is_year_start, ts2.is_year_start)
    (False, True)

* :attr:`pandas.Timestamp.is_year_end`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
    ...   return (ts1.is_year_end, ts2.is_year_end)
    (True, False)

* :attr:`pandas.Timestamp.quarter`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=1)
    ...   return (ts1.quarter, ts2.quarter)
    (4, 3)

* :attr:`pandas.Timestamp.week`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
    ...   return (ts1.week, ts2.week)
    (35, 38)

* :attr:`pandas.Timestamp.weekofyear`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
    ...   return (ts1.weekofyear, ts2.weekofyear)
    (35, 38)

* :attr:`pandas.Timestamp.value`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(12345, unit="ns").value
    12345

* :meth:`pandas.Timestamp.ceil` ``(freq, ambiguous='raise', nonexistent='raise')``

`Supported arguments`:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - argument
     - datatypes
   * - `freq`
     - string

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-10 00:00:00'))

* :meth:`pandas.Timestamp.date` ``()``

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).date()
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 09:57:44.114123'), datetime.date(2021, 12, 9))


* :meth:`pandas.Timestamp.day_name` ``(*args, **kwargs)``

`Supported arguments`: None

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   day_1 = pd.Timestamp(year=2021, month=12, day=9).day_name()
    ...   day_2 = pd.Timestamp(year=2021, month=12, day=10).day_name()
    ...   day_3 = pd.Timestamp(year=2021, month=12, day=11).day_name()
    ...   return (day_1, day_2, day_3)
    ('Thursday', 'Friday', 'Saturday')

* :meth:`pandas.Timestamp.floor`

`Supported arguments`:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - argument
     - datatypes
   * - `freq`
     - string

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 00:00:00'))

* :meth:`pandas.Timestamp.isocalendar`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
    ...   return (ts1, ts2)
    (2021, 49, 4)

* :meth:`pandas.Timestamp.isoformat`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
    ...   return (ts1, ts2)
    '2021-12-09T09:57:44'

* :meth:`pandas.Timestamp.month_name` ``(*args, **kwargs)``

`Supported arguments`: None

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9).month_name()
    'December'

* :meth:`pandas.Timestamp.normalize`

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).normalize()
    ...   return (ts1, ts2)
    Timestamp('2021-12-09 00:00:00')

* :meth:`pandas.Timestamp.round` ``(freq, ambiguous='raise', nonexistent='raise')``

`Supported arguments`:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - argument
     - datatypes
   * - `freq`
     - string

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 12).round()
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 13).round()
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 00:00:00'),Timestamp('2021-12-10 00:00:00'))

* :meth:`pandas.Timestamp.strftime` ``(format)``

`Supported arguments`:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - argument
     - datatypes
   * - `format`
     - string

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9, hour = 12).strftime('%Y-%m-%d %X')
    '2021-12-09 12:00:00'

* :meth:`pandas.Timestamp.toordinal` ``()``

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9).toordinal()
    738133

* :meth:`pandas.Timestamp.weekday` ``()``

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=10)
    ...   return (ts1.weekday(), ts2.weekday())
    (3, 4)

* :classmeth:`pandas.Timedelta.now` ``(tz=None)``

`Supported arguments`: None

`Example Usage`::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp.now()
    Timestamp('2021-12-10 10:54:06.457168')


