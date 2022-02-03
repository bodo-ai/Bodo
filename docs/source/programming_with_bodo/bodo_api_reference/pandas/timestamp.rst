

Timestamp
~~~~~~~~~

Timestamp functionality is documented in `pandas.Timestamp <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html>`_.

``pd.Timestamp``
*****************

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
         - - constant string
       * - ``year``
         - - integer
       * - ``month``
         - - integer
       * - ``day``
         - - integer
       * - ``hour``
         - - integer
       * - ``minute``
         - - integer
       * - ``second``
         - - integer
       * - ``microsecond``
         - - integer
       * - ``nanosecond``
         - - integer

    `Example Usage`:

    .. code-block:: ipython3

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

``pd.Timestamp.day``
^^^^^^^^^^^^^^^^^^^^


* :attr:`pandas.Timestamp.day`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.day
        >>> f()
        9

``pd.Timestamp.hour``
^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.hour`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.hour
        >>> f()
        9

``pd.Timestamp.microsecond``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.microsecond`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.microsecond
        >>> f()
        114123

``pd.Timestamp.month``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.month`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.month
        >>> f()
        month


``pd.Timestamp.nanosecond``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.nanosecond`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(12, unit="ns")
        ...   return ts2.nanosecond
        >>> f()
        12

``pd.Timestamp.second``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.second`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.second
        >>> f()
        44

``pd.Timestamp.year``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.year`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.year
        >>> f()
        2021

``pd.Timestamp.dayofyear``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.dayofyear`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.dayofyear
        >>> f()
        343

``pd.Timestamp.day_of_year``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.day_of_year`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.day_of_year
        >>> f()
        343

``pd.Timestamp.dayofweek``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.dayofweek`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.day_of_year
        >>> f()
        343

``pd.Timestamp.day_of_week``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.day_of_week`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.day_of_week
        >>> f()
        3

``pd.Timestamp.days_in_month``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.days_in_month`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.days_in_month
        >>> f()
        31


``pd.Timestamp.daysinmonth``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.daysinmonth`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return ts2.daysinmonth
        >>> f()
        31

``pd.Timestamp.is_leap_year``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.is_leap_year`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2020, month=2,day=2)
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   return (ts1.is_leap_year, ts2.is_leap_year)
        >>> f()
        (True, False)

``pd.Timestamp.is_month_start``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* :attr:`pandas.Timestamp.is_month_start`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=2)
        ...   return (ts1.is_month_start, ts2.is_month_start)
        >>> f()
        (True, False)

``pd.Timestamp.is_month_end``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.is_month_end`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=30)
        ...   return (ts1.is_month_end, ts2.is_month_end)
        >>> f()
        (True, False)

``pd.Timestamp.is_quarter_start``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.is_quarter_start`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
        ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
        ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
        >>> f()
        (False, True)

``pd.Timestamp.is_quarter_end``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.is_quarter_end`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
        ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
        ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
        >>> f()
        (True, False)

``pd.Timestamp.is_year_start``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.is_year_start`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
        ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
        ...   return (ts1.is_year_start, ts2.is_year_start)
        >>> f()
        (False, True)

``pd.Timestamp.is_year_end``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.is_year_end`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
        ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
        ...   return (ts1.is_year_end, ts2.is_year_end)
        >>> f()
        (True, False)

``pd.Timestamp.quarter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.quarter`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
        ...   ts2 = pd.Timestamp(year=2021, month=9, day=1)
        ...   return (ts1.quarter, ts2.quarter)
        >>> f()
        (4, 3)

``pd.Timestamp.week``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.week`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
        ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
        ...   return (ts1.week, ts2.week)
        >>> f()
        (35, 38)

``pd.Timestamp.weekofyear``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.weekofyear`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
        ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
        ...   return (ts1.weekofyear, ts2.weekofyear)
        >>> f()
        (35, 38)

``pd.Timestamp.value``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :attr:`pandas.Timestamp.value`

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   return pd.Timestamp(12345, unit="ns").value
        >>> f()
        12345

``pd.Timestamp.ceil``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.ceil` ``(freq, ambiguous='raise', nonexistent='raise')``

    `Supported arguments`:

    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - argument
         - datatypes
       * - ``freq``
         - - string

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
        ...   return (ts1, ts2)
        >>> f()
        (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-10 00:00:00'))

``pd.Timestamp.date``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.date` ``()``

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).date()
        ...   return (ts1, ts2)
        >>> f()
        (Timestamp('2021-12-09 09:57:44.114123'), datetime.date(2021, 12, 9))


``pd.Timestamp.day_name``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.day_name` ``(*args, **kwargs)``

    `Supported arguments`: None

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   day_1 = pd.Timestamp(year=2021, month=12, day=9).day_name()
        ...   day_2 = pd.Timestamp(year=2021, month=12, day=10).day_name()
        ...   day_3 = pd.Timestamp(year=2021, month=12, day=11).day_name()
        ...   return (day_1, day_2, day_3)
        >>> f()
        ('Thursday', 'Friday', 'Saturday')

``pd.Timestamp.floor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.floor` ``(freq, ambiguous='raise', nonexistent='raise')``

    `Supported arguments`:

    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - argument
         - datatypes
       * - ``freq``
         - - string

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
        ...   return (ts1, ts2)
        >>> f()
        (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 00:00:00'))

``pd.Timestamp.isocalendar``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.isocalendar` ``()``

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
        ...   return (ts1, ts2)
        >>> f()
        (2021, 49, 4)

``pd.Timestamp.isoformat``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.isoformat` ``()``

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
        ...   return (ts1, ts2)
        >>> f()
        '2021-12-09T09:57:44'

``pd.Timestamp.month_name``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.month_name` ``(locale=None)``

    `Supported arguments`: None

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   return pd.Timestamp(year=2021, month=12, day=9).month_name()
        >>> f()
        'December'

``pd.Timestamp.normalize``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.normalize` ``()``

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).normalize()
        ...   return (ts1, ts2)
        >>> f()
        Timestamp('2021-12-09 00:00:00')

``pd.Timestamp.round``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.round` ``(freq, ambiguous='raise', nonexistent='raise')``

    `Supported arguments`:

    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - argument
         - datatypes
       * - ``freq``
         - - string

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 12).round()
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 13).round()
        ...   return (ts1, ts2)
        >>> f()
        (Timestamp('2021-12-09 00:00:00'),Timestamp('2021-12-10 00:00:00'))


``pd.Timestamp.strftime``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.strftime` ``(format)``

    `Supported arguments`:

    .. list-table::
       :widths: 25 25
       :header-rows: 1

       * - argument
         - datatypes
       * - ``format``
         - - string

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   return pd.Timestamp(year=2021, month=12, day=9, hour = 12).strftime('%Y-%m-%d %X')
        >>> f()
        '2021-12-09 12:00:00'

``pd.Timestamp.toordinal``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.toordinal` ``()``

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   return pd.Timestamp(year=2021, month=12, day=9).toordinal()
        >>> f()
        738133

``pd.Timestamp.weekday``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.weekday` ``()``

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   ts1 = pd.Timestamp(year=2021, month=12, day=9)
        ...   ts2 = pd.Timestamp(year=2021, month=12, day=10)
        ...   return (ts1.weekday(), ts2.weekday())
        >>> f()
        (3, 4)

``pd.Timestamp.now``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.Timestamp.now` ``(tz=None)``

    `Supported arguments`: None

    `Example Usage`:

    .. code-block:: ipython3

        >>> @bodo.jit
        ... def f():
        ...   return pd.Timestamp.now()
        >>> f()
        Timestamp('2021-12-10 10:54:06.457168')


