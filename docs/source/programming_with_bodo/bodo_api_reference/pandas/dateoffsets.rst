

Date Offsets
~~~~~~~~~~~~

Bodo supports a subset of the offset types in ``pandas.tseries.offsets``:

DateOffset
**********


``pd.tseries.offsets.DateOffset``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :func:`pandas.tseries.offsets.DateOffset` ``(n=1, normalize=False, years=None, months=None, weeks=None, days=None, hours=None, minutes=None, seconds=None, microseconds=None, nanoseconds=None, year=None, month=None, day=None, weekday=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None)``

`Supported arguments`:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - default value
     - datatypes
   * - ``n``
     - ``1``
     - integer
   * - ``normalize``
     - ``False``
     - boolean
   * - ``years``
     - ``None``
     - integer
   * - ``months``
     - ``None``
     - integer
   * - ``weeks``
     - ``None``
     - integer
   * - ``days``
     - ``None``
     - integer
   * - ``hours``
     - ``None``
     - integer
   * - ``minutes``
     - ``None``
     - integer
   * - ``seconds``
     - ``None``
     - integer
   * - ``microseconds``
     - ``None``
     - integer
   * - ``nanoseconds``
     - ``None``
     - integer
   * - ``year``
     - ``None``
     - integer
   * - ``month``
     - ``None``
     - integer
   * - ``weekday``
     - ``None``
     - integer
   * - ``day``
     - ``None``
     - integer
   * - ``hour``
     - ``None``
     - integer
   * - ``minute``
     - ``None``
     - integer
   * - ``second``
     - ``None``
     - integer
   * - ``microsecond``
     - ``None``
     - integer
   * - ``nanosecond``
     - ``None``
     - integer

`Example Usage`::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.DateOffset(n=4, normalize=True, weeks=11, hour=2)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
   >>> f(ts)

   Timestamp('2021-09-03 02:00:00')

`Properties`:

``pd.tseries.offsets.DateOffset.normalize``
""""""""""""""""""""""""""""""""""""""""""""""

* :attr:`pandas.tseries.offsets.DateOffset.normalize`

``pd.tseries.offsets.DateOffset.n``
""""""""""""""""""""""""""""""""""""

* :attr:`pandas.tseries.offsets.DateOffset.n`

MonthBegin
**********

``pd.tseries.offsets.MonthBegin``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* :func:`pandas.tseries.offsets.MonthBegin` ``(n=1, normalize=False)``

`Supported arguments`:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - default value
     - datatypes
   * - ``n``
     - ``1``
     - integer
   * - ``normalize``
     - ``False``
     - boolean

`Example Usage`::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.MonthBegin(n=4, normalize=True)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
   >>> f(ts)

   Timestamp('2021-02-01 00:00:00')


MonthEnd
**********

``pd.tseries.offsets.MonthEnd``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :func:`pandas.tseries.offsets.MonthEnd` ``(n=1, normalize=False)``

`Supported arguments`:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - default value
     - datatypes
   * - ``n``
     - ``1``
     - integer
   * - ``normalize``
     - ``False``
     - boolean

`Example Usage`::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.MonthEnd(n=4, normalize=False)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
   >>> f(ts)

   Timestamp('2021-01-31 22:00:00')


Week
****

``pd.tseries.offsets.Week``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :func:`pandas.tseries.offsets.Week` ``(n=1, normalize=False, weekday=None)``

`Supported arguments`:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - default value
     - datatypes
   * - ``n``
     - ``1``
     - integer
   * - ``normalize``
     - ``False``
     - boolean
   * - ``weekday``
     - ``None``
     - integer

`Example Usage`::

   >>> @bodo.jit
   >>> def f(ts):
   ...     return ts + pd.tseries.offsets.Week(n=4, normalize=True, weekday=5)
   >>> ts = pd.Timestamp(year=2020, month=10, day=30, hour=22)
   >>> f(ts)

   Timestamp('2020-11-21 00:00:00')

Binary Operations
******************

For all offsets, addition and subtraction with a scalar
``datetime.date``, ``datetime.datetime`` or ``pandas.Timestamp``
is supported. Multiplication is also supported with a scalar integer.
