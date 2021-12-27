.. _series:

Series
~~~~~~

Bodo provides extensive Series support.
However, operations between Series (+, -, /, *, **) do not
implicitly align values based on their
associated index values yet.


* :class:`pandas.Series`

  * Arguments ``data``, ``index``, and ``name`` are supported.
    ``data`` can be a list, array, Series, Index, or None.
    If ``data`` is Series and ``index`` is provided, implicit alignment is
    not performed yet.


Attributes:

* :attr:`pandas.Series.index`
* :attr:`pandas.Series.values`
* :attr:`pandas.Series.dtype` (object data types such as dtype of
  string series not supported yet)
* :attr:`pandas.Series.shape`
* :attr:`pandas.Series.nbytes`
* :attr:`pandas.Series.ndim`
* :attr:`pandas.Series.size`
* :attr:`pandas.Series.T`
* :meth:`pandas.Series.memory_usage` argument `index` supported
* :attr:`pandas.Series.hasnans`
* :attr:`pandas.Series.empty`
* :attr:`pandas.Series.dtypes`
* :attr:`pandas.Series.name`


Methods:

Conversion:

* :meth:`pandas.Series.astype` (only ``dtype`` argument)
* :meth:`pandas.Series.copy` (including ``deep`` argument)
* :meth:`pandas.Series.infer_objects`
* :meth:`pandas.Series.to_numpy`
* :meth:`pandas.Series.to_list`
* :meth:`pandas.Series.tolist`


Indexing, iteration:

Location based indexing using `[]`, `iat`, and `iloc` is supported.
Changing values of existing string Series using these operators
is not supported yet.

* :meth:`pandas.Series.iat`
* :meth:`pandas.Series.iloc`
* :meth:`pandas.Series.loc`
  Read support for all indexers except using a `callable` object.
  Label-based indexing is not supported yet.

Binary operator functions:

The `fill_value` optional argument for binary functions below is supported.

* :meth:`pandas.Series.add`
* :meth:`pandas.Series.sub`
* :meth:`pandas.Series.mul`
* :meth:`pandas.Series.div`
* :meth:`pandas.Series.truediv`
* :meth:`pandas.Series.floordiv`
* :meth:`pandas.Series.mod`
* :meth:`pandas.Series.pow`
* :meth:`pandas.Series.radd`
* :meth:`pandas.Series.rsub`
* :meth:`pandas.Series.rmul`
* :meth:`pandas.Series.rdiv`
* :meth:`pandas.Series.rtruediv`
* :meth:`pandas.Series.rfloordiv`
* :meth:`pandas.Series.rmod`
* :meth:`pandas.Series.rpow`
* :meth:`pandas.Series.combine`
* :meth:`pandas.Series.round` (`decimals` argument supported)
* :meth:`pandas.Series.lt`
* :meth:`pandas.Series.gt`
* :meth:`pandas.Series.le`
* :meth:`pandas.Series.ge`
* :meth:`pandas.Series.ne`
* :meth:`pandas.Series.eq`
* :meth:`pandas.Series.product`
* :meth:`pandas.Series.dot`

Function application, GroupBy & Window:

* :meth:`pandas.Series.apply` (`convert_dtype` not supported yet)

  - `func` argument can be a function (e.g. lambda), a jit function, or a constant string.
    Constant strings must refer to a supported Series method or Numpy ufunc.

* :meth:`pandas.Series.map` (only the `arg` argument, which should be a function or dictionary)
* :meth:`pandas.Series.groupby` (pass array to `by` argument, or level=0 with regular Index,
  `sort=False` and `observed=True` are set by default)
* :meth:`pandas.Series.rolling` (`window`, `min_periods` and `center` arguments supported)
* :meth:`pandas.Series.pipe` `func` should be a function (not tuple)


Computations / Descriptive Stats:

Statistical functions below are supported without optional arguments
unless support is explicitly mentioned.

* :meth:`pandas.Series.abs`
* :meth:`pandas.Series.all`
* :meth:`pandas.Series.any`
* :meth:`pandas.Series.autocorr` (supports `lag` argument)
* :meth:`pandas.Series.between`
* :meth:`pandas.Series.corr`
* :meth:`pandas.Series.count`
* :meth:`pandas.Series.cov` (supports ddof)
* :meth:`pandas.Series.cummin`
* :meth:`pandas.Series.cummax`
* :meth:`pandas.Series.cumprod`
* :meth:`pandas.Series.cumsum`
* :meth:`pandas.Series.describe` (supports numeric types. Assumes `datetime_is_numeric=True`.)
* :meth:`pandas.Series.diff` (Implemented for Numpy Array data types. Supports `periods` argument.)
* :meth:`pandas.Series.kurt` argument `skipna` supported
* :meth:`pandas.Series.mad` argument `skipna` supported
* :meth:`pandas.Series.max`
* :meth:`pandas.Series.mean`
* :meth:`pandas.Series.median` (supports `skipna` argument)
* :meth:`pandas.Series.min`
* :meth:`pandas.Series.nlargest` (non-numerics not supported yet)
* :meth:`pandas.Series.nsmallest` (non-numerics not supported yet)
* :meth:`pandas.Series.pct_change` (supports numeric types and
  only the `periods` argument supported)
* :meth:`pandas.Series.prod`
* :meth:`pandas.Series.product`
* :meth:`pandas.Series.quantile`
* :meth:`pandas.Series.sem` (support `skipna` and `ddof` arguments)
* :meth:`pandas.Series.skew` argument `skipna` supported
* :meth:`pandas.Series.std` (support `skipna` and `ddof` arguments)
* :meth:`pandas.Series.sum`
* :meth:`pandas.Series.var` (support `skipna` and `ddof` arguments)
* :meth:`pandas.Series.kurtosis` argument `skipna` supported
* :meth:`pandas.Series.unique` the output is assumed to be "small" relative to input and is replicated.
  Use Series.drop_duplicates() if the output should remain distributed.
* :meth:`pandas.Series.nunique` all optional arguments are supported
* :attr:`pandas.Series.is_monotonic`
* :attr:`pandas.Series.is_monotonic_increasing`
* :attr:`pandas.Series.is_monotonic_decreasing`
* :meth:`pandas.Series.value_counts` all optional arguments except `dropna` are supported.


Reindexing / Selection / Label manipulation:


* :meth:`pandas.Series.drop_duplicates`
* :meth:`pandas.Series.equals` (series and `other` should contain scalar values in each row)
* :meth:`pandas.Series.head` (`n` argument is supported)
* :meth:`pandas.Series.idxmax`
* :meth:`pandas.Series.idxmin`
* :meth:`pandas.Series.isin`
  `values` argument supports both distributed array/Series and replicated list/array/Series
* :meth:`pandas.Series.rename` (only set a new name using a string value)
* :meth:`pandas.Series.reset_index` For MultiIndex case, only dropping all levels supported.
  Requires Index name to be known at compilation time if `drop=False`.
* :meth:`pandas.Series.take`
* :meth:`pandas.Series.tail` (`n` argument is supported)
* :meth:`pandas.Series.where` (`cond` and `other` arguments supported for 1d numpy data arrays. Categorical data supported for scalar 'other'.)
* :meth:`pandas.Series.mask` (`cond` and `other` arguments supported for 1d numpy data arrays. Categorical data supported for scalar 'other'.)

Missing data handling:

* :meth:`pandas.Series.backfill`
* :meth:`pandas.Series.bfill`
* :meth:`pandas.Series.dropna`
* :meth:`pandas.Series.ffill`
* :meth:`pandas.Series.fillna`
* :meth:`pandas.Series.isna`
* :meth:`pandas.Series.isnull`
* :meth:`pandas.Series.notna`
* :meth:`pandas.Series.notnull`
* :meth:`pandas.Series.pad`
* :meth:`pandas.Series.replace`

Reshaping, sorting:

* :meth:`pandas.Series.argsort`
* :meth:`pandas.Series.sort_values` `ascending` and `na_position` arguments are supported
* :meth:`pandas.Series.sort_index` `ascending` and `na_position` arguments are supported
* :meth:`pandas.Series.explode`
* :meth:`pandas.Series.repeat`

Combining / comparing / joining / merging:

* :meth:`pandas.Series.append` `ignore_index` is supported.
  setting name for output Series not supported yet)

Time series-related:

* :meth:`pandas.Series.shift` (supports numeric, boolean, datetime.date, datetime64, timedelta64, and string types.
  Only the `periods` argument is supported)

Datetime properties:

* :attr:`pandas.Series.dt.date`
* :attr:`pandas.Series.dt.year`
* :attr:`pandas.Series.dt.month`
* :attr:`pandas.Series.dt.day`
* :attr:`pandas.Series.dt.hour`
* :attr:`pandas.Series.dt.minute`
* :attr:`pandas.Series.dt.second`
* :attr:`pandas.Series.dt.microsecond`
* :attr:`pandas.Series.dt.nanosecond`
* :attr:`pandas.Series.dt.week`
* :attr:`pandas.Series.dt.weekofyear`
* :attr:`pandas.Series.dt.day_of_week`
* :attr:`pandas.Series.dt.weekday`
* :attr:`pandas.Series.dt.dayofyear`
* :attr:`pandas.Series.dt.day_of_year`
* :attr:`pandas.Series.dt.quarter`
* :attr:`pandas.Series.dt.is_month_start`
* :attr:`pandas.Series.dt.is_month_end`
* :attr:`pandas.Series.dt.is_quarter_start`
* :attr:`pandas.Series.dt.is_quarter_end`
* :attr:`pandas.Series.dt.is_year_start`
* :attr:`pandas.Series.dt.is_year_end`
* :attr:`pandas.Series.dt.daysinmonth`
* :attr:`pandas.Series.dt.days_in_month`

Datetime methods:

* :meth:`pandas.Series.dt.normalize`
* :meth:`pandas.Series.dt.strftime`
* :meth:`pandas.Series.dt.round`
* :meth:`pandas.Series.dt.floor`
* :meth:`pandas.Series.dt.ceil`
* :meth:`pandas.Series.dt.month_name` (``locale`` not supported)
* :meth:`pandas.Series.dt.day_name` (``locale`` not supported)

String handling:

* :meth:`pandas.Series.str.capitalize`
* :meth:`pandas.Series.str.center`
* :meth:`pandas.Series.str.contains` `regex` and `case` arguments supported.
* :meth:`pandas.Series.str.count`
* :meth:`pandas.Series.str.endswith`
* :meth:`pandas.Series.str.extract` (input pattern should be a constant string)
* :meth:`pandas.Series.str.extractall` (input pattern should be a constant string)
* :meth:`pandas.Series.str.find`
* :meth:`pandas.Series.str.get`
* :meth:`pandas.Series.str.join`
* :meth:`pandas.Series.str.len`
* :meth:`pandas.Series.str.ljust`
* :meth:`pandas.Series.str.lower`
* :meth:`pandas.Series.str.lstrip` `to_strip` argument supported.
* :meth:`pandas.Series.str.pad`
* :meth:`pandas.Series.str.repeat`
* :meth:`pandas.Series.str.replace` `regex` argument supported.
* :meth:`pandas.Series.str.rfind`
* :meth:`pandas.Series.str.rjust`
* :meth:`pandas.Series.str.rstrip` `to_strip` argument supported.
* :meth:`pandas.Series.str.slice`
* :meth:`pandas.Series.str.slice_replace`
* :meth:`pandas.Series.str.split`
* :meth:`pandas.Series.str.startswith`
* :meth:`pandas.Series.str.strip` `to_strip` argument supported.
* :meth:`pandas.Series.str.swapcase`
* :meth:`pandas.Series.str.title`
* :meth:`pandas.Series.str.upper`
* :meth:`pandas.Series.str.zfill`
* :meth:`pandas.Series.str.isalnum`
* :meth:`pandas.Series.str.isalpha`
* :meth:`pandas.Series.str.isdigit`
* :meth:`pandas.Series.str.isspace`
* :meth:`pandas.Series.str.islower`
* :meth:`pandas.Series.str.isupper`
* :meth:`pandas.Series.str.istitle`
* :meth:`pandas.Series.str.isnumeric`
* :meth:`pandas.Series.str.isdecimal`

Categorical accessor:


* :attr:`pandas.Series.cat.codes`

Serialization / IO / Conversion

* :meth:`pandas.Series.to_csv`
* :meth:`pandas.Series.to_dict` is not parallelized since dictionaries are not parallelized
* :meth:`pandas.Series.to_frame` Series name should be a known constant or a constant 'name' should be provided

.. _heterogeneous_series:

Heterogeneous Series
~~~~~~~~~~~~~~~~~~~~

Bodo's Series implementation requires all elements to share a common data type.
However, in situations where the size and types of the elements are constant at
compile time, Bodo has some mixed type handling with its Heterogeneous Series type.

.. warning::

  This type's primary purpose is for iterating through the rows of a DataFrame
  with different column types. You should not attempt to directly create Series
  with mixed types.

Heterogeneous Series operations are a subset of those supported for Series and
the supported operations are listed below. Please refer to :ref:`series` for
detailed usage.

Attributes:

* :attr:`pandas.Series.index`
* :attr:`pandas.Series.values`
* :attr:`pandas.Series.shape`
* :attr:`pandas.Series.nbytes`
* :attr:`pandas.Series.ndim`
* :attr:`pandas.Series.size`
* :attr:`pandas.Series.T`
* :attr:`pandas.Series.empty`
* :attr:`pandas.Series.name`


Methods:

* :meth:`pandas.Series.copy`

