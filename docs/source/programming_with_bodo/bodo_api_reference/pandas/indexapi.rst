
Index objects
~~~~~~~~~~~~~

Index
*****

Properties

* :attr:`pandas.Index.name`

 Example Usage::

     >>> @bodo.jit
     ... def f(I):
     ...   return I.name

     >>> I = pd.Index([1,2,3], name = "hello world")
     >>> f(I)
     "hello world"

* :attr:`pandas.Index.shape`

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

 Example Usage::

     >>> @bodo.jit
     ... def f(I):
     ...   return I.shape

     >>> I = pd.Index([1,2,3])
     >>> f(I)
     (3,)

* :attr:`pandas.Index.values`

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

  Example Usage::

     >>> @bodo.jit
     ... def f(I):
     ...   return I.values

     >>> I = pd.Index([1,2,3])
     >>> f(I)
     [1 2 3]
* :attr:`pandas.Index.nbytes`

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

 Important Notes:
  Currently, bodo upcasts all numeric index data types to 64 bitwidth.

 Example Usage::

    >>> @bodo.jit
    ... def f(I):
    ...   return I.nbytes

    >>> I1 = pd.Index([1,2,3,4,5,6], dtype = np.int64)
    >>> 48
    >>> I2 = pd.Index([1,2,3], dtype = np.int64)
    >>> 24
    >>> I3 = pd.Index([1,2,3], dtype = np.int32)
    >>> 24



Modifying and computations:

* :meth:`pandas.Index.copy` ``(name=None, deep=False, dtype=None, names=None)``

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

 Supported Arguments:
  * name


 Example Usage::

    >>> @bodo.jit
    ... def f(I):
    ...   return I.copy(name="new_name")

    >>> I = pd.Index([1,2,3], name = "origial_name")
    >>> f(I)
    Int64Index([1, 2, 3], dtype='int64', name='new_name')

* :meth:`pandas.Index.get_loc` ``(key, method=None, tolerance=None)``
  Returns the location of the specified index label.

  (Should be about as fast as standard python, maybe slightly slower)

 Unsupported Index Types:
  * CategoricalIndex
  * MultiIndex
  * IntervalIndex

 Supported Arguments:
  * key (must be of same type as the index)

 Important Notes:
  * Only works for index with unique values (scalar return).
  * Only works with replicated Index


  Example Usage::

    >>> @bodo.jit
    ... def f(I):
    ...   return I.get_loc(2)

    >>> I = pd.Index([1,2,3])
    >>> f(I)
    1

* :meth:`pandas.Index.take` ``(indices, axis=0, allow_fill=True, fill_value=None, **kwargs)``
  Return a new Index of the values selected by the indices.

 Supported Arguments:
    * indicies
      - can be boolean Array like, integer Array like, or slice

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

 Important Notes:
  * Bodo **Does Not** support ``kwargs``, even for compatibility.

* :meth:`pandas.Index.min` ``(axis=None, skipna=True, *args, **kwargs)``

  Supported arguments:
    None

  **Supported** Index Types:
    * TimedeltaIndex
    * DatetimeIndex

 Important Notes:
  * Bodo **Does Not** support ``args`` and ``kwargs``, even for compatibility.
  * For DatetimeIndex, will throw an error if all values in the index are null.

    Example Usage::

    >>> @bodo.jit
    ... def f(I):
    ...   return I.min()

    >>> I = pd.Index(pd.date_range(start="2018-04-24", end="2018-04-25", periods=5))
    >>> f(I)
    2018-04-24 00:00:00


* :meth:`pandas.Index.max` ``(axis=None, skipna=True, *args, **kwargs)``

 Supported arguments:
  None

 **Supported** Index Types:
  * TimedeltaIndex
  * DatetimeIndex

 Important Notes:
  * Bodo **Does Not** support ``args`` and ``kwargs``, even for compatibility.
  * For DatetimeIndex, will throw an error if all values in the index are null.


Example Usage::

  >>> @bodo.jit
  ... def f(I):
  ...   return I.min()

  >>> I = pd.Index(pd.date_range(start="2018-04-24", end="2018-04-25", periods=5))
  >>> f(I)
  2018-04-25 00:00:00


Missing values:

* :meth:`pandas.Index.isna` ``()``

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.isna()

    >>> I = pd.Index([1,None,3])
    >>> f(I)
    [False  True False]

* :meth:`pandas.Index.isnull` ``()``

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.isnull()

    >>> I = pd.Index([1,None,3])
    >>> f(I)
    [False  True False]

Conversion:

* :meth:`pandas.Index.map` ``(mapper, na_action=None)``

 Unsupported Index Types:
  * MultiIndex
  * IntervalIndex

Supported arguments:
  * mapper
    - must be a function, function cannot return tuple type

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.map(lambda x: x + 2)

    >>> I = pd.Index([1,None,3])
    >>> f(I)
    Float64Index([3.0, nan, 5.0], dtype='float64')


Numeric Index
*************

Numeric index objects ``RangeIndex``, ``Int64Index``, ``UInt64Index`` and
``Float64Index`` are supported as index to dataframes and series.
Constructing them in Bodo functions, passing them to Bodo functions (unboxing),
and returning them from Bodo functions (boxing) are also supported.

* :class:`pandas.RangeIndex` ``(start=None, stop=None, step=None, dtype=None, copy=False, name=None)``


Supported arguments:
 * start
   - Integer
 * stop
   - integer
 * step
    - Integer
 * name
   - String


  Example Usage::
    >>> @bodo.jit
    ... def f():
    ...   return pd.RangeIndex(0, 10, 2)

    >>> f(I)
    RangeIndex(start=0, stop=10, step=2)



* :class:`pandas.Int64Index` ``(data=None, dtype=None, copy=False, name=None)``
* :class:`pandas.UInt64Index` ``(data=None, dtype=None, copy=False, name=None)``
* :class:`pandas.Float64Index` ``(data=None, dtype=None, copy=False, name=None)``

 Supported arguments:
  * data
    - list or array
  * copy
    - Boolean
  * name
    - String


  Example Usage::
    >>> @bodo.jit
    ... def f():
    ... return (pd.Int64Index(np.arange(3)), pd.UInt64Index([1,2,3]), pd.Float64Index(np.arange(3)))

    >>> f()
    (Int64Index([0, 1, 2], dtype='int64'), UInt64Index([0, 1, 2], dtype='uint64'), Float64Index([0.0, 1.0, 2.0], dtype='float64'))



DatetimeIndex
*************

``DatetimeIndex`` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

* :class:`pandas.DatetimeIndex`

 Supported arguments:
  * data
    - array-like of datetime64, Integer, or strings


Date fields of DatetimeIndex are supported:

* :attr:`pandas.DatetimeIndex.year`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.year

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([2019, 2019, 2019, 2020, 2020], dtype='int64')


* :attr:`pandas.DatetimeIndex.month`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.month

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([12, 12, 12, 1, 1], dtype='int64')
* :attr:`pandas.DatetimeIndex.day`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.day

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([31, 31, 31, 1, 1], dtype='int64')
* :attr:`pandas.DatetimeIndex.hour`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.hour

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([2, 12, 22, 9, 19], dtype='int64')

* :attr:`pandas.DatetimeIndex.minute`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.minute

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([32, 42, 52, 2, 12], dtype='int64')

* :attr:`pandas.DatetimeIndex.second`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.second

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([45, 35, 25, 15, 5], dtype='int64')

* :attr:`pandas.DatetimeIndex.microsecond`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.microsecond

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 01:01:01", end="2019-12-31 01:01:02", periods=5))
    >>> f(I)
    Int64Index([0, 250000, 500000, 750000, 0], dtype='int64')


* :attr:`pandas.DatetimeIndex.nanosecond`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.nanosecond

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 01:01:01.0000001", end="2019-12-31 01:01:01.0000002", periods=5))
    >>> f(I)
    Int64Index([100, 125, 150, 175, 200], dtype='int64')

* :attr:`pandas.DatetimeIndex.date`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.date

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    [datetime.date(2019, 12, 31) datetime.date(2019, 12, 31) datetime.date(2019, 12, 31) datetime.date(2020, 1, 1) datetime.date(2020, 1, 1)]

* :attr:`pandas.DatetimeIndex.dayofyear`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.dayofyear

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([365, 365, 365, 1, 1], dtype='int64')


* :attr:`pandas.DatetimeIndex.day_of_year`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.day_of_year

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([365, 365, 365, 1, 1], dtype='int64')


* :attr:`pandas.DatetimeIndex.dayofweek`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.dayofweek

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 2, 2], dtype='int64')


* :attr:`pandas.DatetimeIndex.day_of_week`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.day_of_week

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 2, 2], dtype='int64')


* :attr:`pandas.DatetimeIndex.is_leap_year`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_leap_year

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    [Flase False False True True]

* :attr:`pandas.DatetimeIndex.is_month_start`

Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_month_start

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([0, 0, 0, 1, 1], dtype='int64')

* :attr:`pandas.DatetimeIndex.is_month_end`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_month_end

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 0, 0], dtype='int64')

* :attr:`pandas.DatetimeIndex.is_quarter_start`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_quarter_start

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([0, 0, 0, 1, 1], dtype='int64')

* :attr:`pandas.DatetimeIndex.is_quarter_end`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_quarter_end

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 0, 0], dtype='int64')

* :attr:`pandas.DatetimeIndex.is_year_start`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_year_start

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([0, 0, 0, 1, 1], dtype='int64')

* :attr:`pandas.DatetimeIndex.is_year_end`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.is_year_end

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 0, 0], dtype='int64')
* :attr:`pandas.DatetimeIndex.week`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.week

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 1, 1], dtype='int64')

* :attr:`pandas.DatetimeIndex.weekday`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.weekday

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 2, 2], dtype='int64')

* :attr:`pandas.DatetimeIndex.weekofyear`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.weekofyear

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([1, 1, 1, 1,1], dtype='int64')

* :attr:`pandas.DatetimeIndex.quarter`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.quarter

    >>> I = pd.DatetimeIndex(pd.date_range(start="2019-12-31 02:32:45", end="2020-01-01 19:12:05", periods=5))
    >>> f(I)
    Int64Index([4, 4, 4, 1, 1], dtype='int64')



Subtraction of ``Timestamp`` from ``DatetimeIndex`` and vice versa
is supported.

Comparison operators ``==``, ``!=``, ``>=``, ``>``, ``<=``, ``<`` between
``DatetimeIndex`` and a string of datetime
are supported.


TimedeltaIndex
**************

``TimedeltaIndex`` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

* :class:`pandas.TimedeltaIndex` ``(data=None, unit=None, freq=NoDefault.no_default, closed=None, dtype=dtype('<m8[ns]'), copy=False, name=None)``

 Supported arguments:
  * data
   - must be array-like of timedelta64ns or Ingetger.

Time fields of TimedeltaIndex are supported:

* :attr:`pandas.TimedeltaIndex.days`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.days

    >>> I = pd.TimedeltaIndex([pd.Timedelta(3, unit="D"))])
    >>> f(I)
    Int64Index([3], dtype='int64')

* :attr:`pandas.TimedeltaIndex.seconds`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.seconds

    >>> I = pd.TimedeltaIndex([pd.Timedelta(-2, unit="S"))])
    >>> f(I)
    Int64Index([-2], dtype='int64')
* :attr:`pandas.TimedeltaIndex.microseconds`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.microseconds

    >>> I = pd.TimedeltaIndex([pd.Timedelta(11, unit="micros"))])
    >>> f(I)
    Int64Index([11], dtype='int64')
* :attr:`pandas.TimedeltaIndex.nanoseconds`
Example Usage::
    >>> @bodo.jit
    ... def f(I):
    ...   return I.nanoseconds

    >>> I = pd.TimedeltaIndex([pd.Timedelta(7, unit="nanos"))])
    >>> f(I)
    Int64Index([7], dtype='int64')


PeriodIndex
***********

``PeriodIndex`` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.

BinaryIndex
***********

``BinaryIndex`` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.


MultiIndex
**********

* :meth:`pandas.MultiIndex.from_product` (*iterables* and *names* supported as tuples, no parallel support yet)
