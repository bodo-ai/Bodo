

Date Offsets
~~~~~~~~~~~~

Bodo supports a subset of the offset types in ``pandas.tseries.offsets``:

DateOffset
~~~~~~~~~~

* :func:`pandas.tseries.offsets.DateOffset` ``(n=1, normalize=False, years=None, months=None, weeks=None, days=None, hours=None, minutes=None, seconds=None, microseconds=None, nanoseconds=None, year=None, month=None, day=None, weekday=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None)``

Supported Arguments:

  * n (default=1):

    - Integer

  * normalize (default=False):

    - Boolean

  * years (default=None):

    - integer

  * months (default=None):

    - integer

  * weeks (default=None):

    - integer

  * days (default=None):

    - integer

  * hours (default=None):

    - integer

  * minutes (default=None):

    - integer

  * seconds (default=None):

    - integer

  * microseconds (default=None):

    - integer

  * nanoseconds (default=None):

    - integer

  * year (default=None):

    - integer

  * month (default=None):

    - integer

  * weekday (default=None):

    - integer

  * day (default=None):

    - integer

  * hour (default=None):

    - integer

  * minute (default=None):

    - integer

  * second (default=None):

    - integer

  * microsecond (default=None):

    - integer

  * nanosecond (default=None):

    - integer


Example Usage::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.DateOffset(n=4, normalize=True, weeks=11, hour=2)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
   >>> f(ts)

   Timestamp('2021-09-03 02:00:00')

Properties

* :attr:`pandas.tseries.offsets.DateOffset.normalize`
* :attr:`pandas.tseries.offsets.DateOffset.n`

MonthBegin
~~~~~~~~~~

* :func:`pandas.tseries.offsets.MonthBegin` ``(n=1, normalize=False)``

Supported Arguments:

  * n (default=1):

    - Integer

  * normalize (default=False):

    - Boolean


Example Usage::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.MonthBegin(n=4, normalize=True)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
   >>> f(ts)

   Timestamp('2021-02-01 00:00:00')

MonthEnd
~~~~~~~~

* :func:`pandas.tseries.offsets.MonthEnd` ``(n=1, normalize=False)``

Supported Arguments:

  * n (default=1):

    - Integer

  * normalize (default=False):

    - Boolean

Example Usage::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.MonthEnd(n=4, normalize=False)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)

   Timestamp('2021-01-31 22:00:00')


Week
~~~~

* :func:`pandas.tseries.offsets.Week` ``(n=1, normalize=False, weekday=None)``

Supported Arguments:

  * n (default=1):

    - Integer

  * normalize (default=False):

    - Boolean

  * weekday (default=None):

    - integer

Example Usage::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.Week(n=4, normalize=True, weekday=5)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)

   Timestamp('2020-11-21 00:00:00')

Binary Operations
~~~~~~~~~~~~~~~~~

For all offsets, addition and substraction with a scalar
``datetime.date``, ``datetime.datetime`` or ``pandas.Timestamp``
is supported. Multiplication is also supported with a scalar integer.
