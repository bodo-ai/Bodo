
Timedelta
~~~~~~~~~
Timedelta functionality is documented in `pandas.Timedelta <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html>`_.

* :class:`pandas.Timedelta` ``(value=<object object>, unit="ns", days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)``

Supported arguments:
  * value
   - Integer (with constant string unit argument)
   - String
   - pandas Timedelta
   - datetime Timedelta
  * unit
   - Constant String. Only has an effect when passing an integer value, see `here <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html>`_ for allowed values.
  * days
   - Integer
  * seconds
   - Integer
  * microseconds
   - Integer
  * milliseconds
   - Integer
  * minutes
   - Integer
  * hours
   - Integer
  * weeks
   - Integer

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   td1 = pd.Timedelta("10 Seconds")
    ...   td2 = pd.Timedelta(10, unit= "W")
    ...   td3 = pd.Timedelta(days= 10, hours=2, microseconds= 23)
    ...   return (td1, td2, td3)
    (Timedelta('0 days 00:00:10'), Timedelta('70 days 00:00:00'), Timedelta('10 days 02:00:00.000023'))


* :attr:`pandas.Timedelta.components`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).components
    Components(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23, nanoseconds=0)
* :attr:`pandas.Timedelta.days`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).days
    10

* :attr:`pandas.Timedelta.delta`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(microseconds=23).delta
    23000

* :attr:`pandas.Timedelta.microseconds`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).microseconds
    23

* :attr:`pandas.Timedelta.nanoseconds`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).nanoseconds
    0

* :attr:`pandas.Timedelta.seconds`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta("10 nanoseconds").nanoseconds
    10
* :attr:`pandas.Timedelta.value`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta("13 nanoseconds").value
    13

* :meth:`pandas.Timedelta.ceil` ``(freq)``
  Supported arguments:
    * freq
     - String

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).ceil("D")
    11 days 00:00:00

* :meth:`pandas.Timedelta.floor`
  Supported arguments:
    * freq
     - String

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).floor("D")
    10 days 00:00:00
* :meth:`pandas.Timedelta.round`
  Supported arguments:
    * freq
     - String

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return (pd.Timedelta(days=10, hours=12).round("D"), pd.Timedelta(days=10, hours=13).round("D"))
    (Timedelta('10 days 00:00:00'), Timedelta('11 days 00:00:00'))

* :meth:`pandas.Timedelta.to_numpy` ``()``

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_numpy()
    871623013023000 nanoseconds
* :meth:`pandas.Timedelta.to_pytimedelta` ``()``

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_pytimedelta()
    10 days, 2:07:03.013023

* :meth:`pandas.Timedelta.to_timedelta64` ``()``

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).to_timedelta64()
    871623013023000 nanoseconds
* :meth:`pandas.Timedelta.total_seconds` ``()``

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timedelta(days=10, hours=2, minutes=7, seconds=3, milliseconds=13, microseconds=23).total_seconds()
    871623.013023
