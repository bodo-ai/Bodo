Supported Pandas Operations
---------------------------

Below is the list of the Pandas operators that Bodo supports.
Optional arguments are not supported unless if specified.
Since Numba doesn't support Pandas, only these operations
can be used for both large and small datasets.

In addition:

* Accessing columns using both getitem (e.g. ``df['A']``) and attribute
  (e.g. ``df.A``) is supported.
* Using columns similar to Numpy arrays and performing data-parallel operations
  listed previously is supported.
* Filtering data frames using boolean arrays is supported
  (e.g. ``df[df.A > .5]``).


Integer NaN Issue
~~~~~~~~~~~~~~~~~

DataFrame columns with integer data need special care. Pandas dynamically
converts integer columns to floating point when NaN values are needed.
This is because Numpy does not support NaN values for integers.
Bodo does not perform this conversion unless enough information is
available at compilation time. Hence, the user is responsible for manual
conversion of integer data to floating point data if needed.

Input/Output
~~~~~~~~~~~~

* :func:`pandas.read_csv`

   * Arguments ``filepath_or_buffer``, ``sep``, ``delimiter``, ``header``, ``names``,
     ``index_col``, ``usecols``, ``dtype``, ``skiprows`` and ``parse_dates`` are supported.
   * ``filepath_or_buffer`` should be a string and is required.
   * Either ``names`` and ``dtype`` arguments should be provided to enable type inference,
     or ``filepath_or_buffer`` should be a constant string for Bodo to infer types by looking at the file at compile time.
   * ``names``, ``usecols``, ``parse_dates`` should be constant lists.
   * ``dtype`` should be a constant dictionary of strings and types.

* :func:`pandas.read_parquet`

   * Arguments ``path`` and ``columns`` are supported. ``columns``
     should be a constant list of strings.

   * If ``path`` is constant, Bodo finds the schema from file at compilation time.
     Otherwise, schema should be provided. For example::

      @bodo.jit(locals={'df':{'A': bodo.float64[:],
                              'B': bodo.string_array_type}})
      def impl(f):
        df = pd.read_parquet(f)
        return df


General functions
~~~~~~~~~~~~~~~~~

Data manipulations:

* :func:`pandas.crosstab`
   * Annotation of pivot values is required.
     For example, `@bodo.jit(pivots={'pt': ['small', 'large']})` declares
     the output table `pt` will have columns called `small` and `large`.

* :func:`pandas.merge`

   * Arguments ``left``, ``right`` should be dataframes.
   * `how`, `on`, `left_on`, `right_on`, `left_index`,
     and `right_index` are supported but should be constant values.

* :func:`pandas.merge_asof` (similar arguments to `merge`)

* :func:`pandas.concat`
  Input list or tuple of dataframes or series is supported.


Top-level missing data:

* :func:`pandas.isna`
* :func:`pandas.isnull`
* :func:`pandas.notna`
* :func:`pandas.notnull`


Top-level conversions:

* :func:`pandas.to_numeric` Input can be a Series.
  Output requires type annotation. `errors='coerce'` required.


Top-level dealing with datetimelike:


* :func:`pandas.date_range`

   * ``start``, ``end``, ``periods``, ``freq``, ``name`` and ``closed``
     arguments are supported. This function is not parallelized yet.


Series
~~~~~~

Bodo provides extensive Series support.
However, operations between Series (+, -, /, *, **) do not
implicitly align values based on their
associated index values yet.


* :func:`pandas.Series`

   * Arguments ``data``, ``index``, ``name`` and ``copy`` are supported.
     ``data`` is required and can be a list, array, Series or Index.
     If ``data`` is Series and ``index`` is provided, implicit alignment is
     not performed yet.


Attributes:

* :attr:`Series.index`
* :attr:`Series.values`
* :attr:`Series.dtype` (object data types such as dtype of
  string series not supported yet)
* :attr:`Series.shape`
* :attr:`Series.ndim`
* :attr:`Series.size`
* :attr:`Series.T`
* :attr:`Series.hasnans`
* :attr:`Series.empty`
* :attr:`Series.dtypes`
* :attr:`Series.name`
* :meth:`Series.put` (only numeric data types)


Methods:

Conversion:

* :meth:`Series.astype` (only ``dtype`` argument,
  can be a Numpy numeric dtype or ``str``)
* :meth:`Series.copy` (including ``deep`` argument)
* :meth:`Series.to_list`
* :meth:`Series.get_values`


Indexing, iteration:

Location based indexing using `[]`, `iat`, and `iloc` is supported.
Changing values of existing string Series using these operators
is not supported yet.

* :attr:`Series.iat`
* :attr:`Series.iloc`


Binary operator functions:

The `fill_value` optional argument for binary functions below is supported.

* :meth:`Series.add`
* :meth:`Series.sub`
* :meth:`Series.mul`
* :meth:`Series.div`
* :meth:`Series.truediv`
* :meth:`Series.floordiv`
* :meth:`Series.mod`
* :meth:`Series.pow`
* :meth:`Series.combine`
* :meth:`Series.lt`
* :meth:`Series.gt`
* :meth:`Series.le`
* :meth:`Series.ge`
* :meth:`Series.ne`

Function application, GroupBy & Window:

* :meth:`Series.apply` (only the `func` argument)
* :meth:`Series.map` (only the `arg` argument, which should be a function)
* :meth:`Series.rolling` (`window` and `center` arguments supported)


Computations / Descriptive Stats:

Statistical functions below are supported without optional arguments
unless support is explicitly mentioned.

* :meth:`Series.abs`
* :meth:`Series.corr`
* :meth:`Series.count`
* :meth:`Series.cov`
* :meth:`Series.cumsum`
* :meth:`Series.cumprod`
* :meth:`Series.describe` currently returns a string instead of Series object.
* :meth:`Series.max`
* :meth:`Series.mean`
* :meth:`Series.median`
* :meth:`Series.min`
* :meth:`Series.nlargest` (non-numerics not supported yet)
* :meth:`Series.nsmallest` (non-numerics not supported yet)
* :meth:`Series.pct_change`(supports numeric types and
  only the `periods` argument supported)
* :meth:`Series.prod`
* :meth:`Series.quantile`
* :meth:`Series.std`
* :meth:`Series.sum`
* :meth:`Series.var`
* :meth:`Series.unique`
* :meth:`Series.nunique`
* :meth:`Series.value_counts`


Reindexing / Selection / Label manipulation:


* :meth:`Series.head` (`n` argument is supported)
* :meth:`Series.idxmax`
* :meth:`Series.idxmin`
* :meth:`Series.rename` (only set a new name using a string value)
* :meth:`Series.tail` (`n` argument is supported)
* :meth:`Series.take`

Missing data handling:

* :meth:`Series.isna`
* :meth:`Series.notna`
* :meth:`Series.dropna`
* :meth:`Series.fillna`

Reshaping, sorting:

* :meth:`Series.argsort`
* :meth:`Series.sort_values` (does not push NAs to first/last positions yet)
* :meth:`Series.append` `ignore_index` is supported.
  setting name for output Series not supported yet)

Time series-related:

* :meth:`Series.shift` (supports numeric types and
  only the `periods` argument supported)

String handling:

* :meth:`Series.str.contains`
* :meth:`Series.str.len`

DataFrame
~~~~~~~~~

Bodo provides extensive DataFrame support documented below.


* :func:`pandas.DataFrame`

   ``data`` argument can be a constant dictionary or 2d Numpy array.
   Other arguments are also supported.

Attributes and underlying data:


* :attr:`DataFrame.index` (can access but not set new index yet)
* :attr:`DataFrame.columns`  (can access but not set new columns yet)
* :attr:`DataFrame.values` (only for numeric dataframes)
* :meth:`DataFrame.get_values` (only for numeric dataframes)
* :attr:`DataFrame.ndim`
* :attr:`DataFrame.size`
* :attr:`DataFrame.shape`
* :attr:`DataFrame.empty`

Conversion:

* :meth:`DataFrame.astype` (only accepts a single data type
  of Numpy dtypes or `str`)
* :meth:`DataFrame.copy` (including `deep` flag)
* :meth:`DataFrame.isna`
* :meth:`DataFrame.notna`


Indexing, iteration:

* :meth:`DataFrame.head` (including `n` argument)
* :attr:`DataFrame.iat`
* :attr:`DataFrame.iloc`
* :meth:`DataFrame.tail` (including `n` argument)
* :meth:`DataFrame.isin` (`values` can be a dataframe with matching index
  or a list or a set)

Function application, GroupBy & Window:

* :meth:`DataFrame.apply`
* :meth:`DataFrame.groupby`
* :meth:`DataFrame.rolling`

Computations / Descriptive Stats:

* :meth:`DataFrame.abs`
* :meth:`DataFrame.corr` (`min_periods` argument supported)
* :meth:`DataFrame.count`
* :meth:`DataFrame.cov` (`min_periods` argument supported)
* :meth:`DataFrame.cumprod`
* :meth:`DataFrame.cumsum`
* :meth:`DataFrame.describe`
* :meth:`DataFrame.max`
* :meth:`DataFrame.mean`
* :meth:`DataFrame.median`
* :meth:`DataFrame.min`
* :meth:`DataFrame.pct_change`
* :meth:`DataFrame.prod`
* :meth:`DataFrame.quantile`
* :meth:`DataFrame.sum`
* :meth:`DataFrame.std`
* :meth:`DataFrame.var`
* :meth:`DataFrame.nunique`


Reindexing / Selection / Label manipulation:

* :meth:`DataFrame.drop` (only dropping columns supported,
  either using `columns` argument or setting `axis=1`)
* :meth:`DataFrame.head` (including `n` argument)
* :meth:`DataFrame.idxmax`
* :meth:`DataFrame.idxmin`
* :meth:`DataFrame.reset_index` (only `drop=True` supported)
* :meth:`DataFrame.set_index` `keys` can only be a column label
  (a constant string).
* :meth:`DataFrame.tail` (including `n` argument)
* :meth:`DataFrame.take`

Missing data handling:

* :meth:`DataFrame.dropna`
* :meth:`DataFrame.fillna`

Reshaping, sorting, transposing:

* :meth:`DataFrame.pivot_table`

   * Arguments ``values``, ``index``, ``columns`` and ``aggfunc`` are
     supported.
   * Annotation of pivot values is required.
     For example, `@bodo.jit(pivots={'pt': ['small', 'large']})` declares
     the output pivot table `pt` will have columns called `small` and `large`.

* :meth:`DataFrame.sort_values` `by` argument should be constant string or
  constant list of strings. `ascending` argument is supported.
* :meth:`DataFrame.sort_index` `ascending` argument is supported.

Combining / joining / merging:

* :meth:`DataFrame.append` appending a dataframe or list of dataframes
  supported. `ignore_index=True` is necessary and set by default.
* :meth:`DataFrame.join` only dataframes.
* :meth:`DataFrame.merge` only dataframes. `how`, `on`, `left_on`,
  `right_on`, `left_index`, and `right_index` are supported but
  should be constant values.

Time series-related:

* :meth:`DataFrame.shift` (supports numeric types and
  only the `periods` argument supported)


Numeric Index
~~~~~~~~~~~~~

Numeric index objects ``RangeIndex``, ``Int64Index``, ``UInt64Index`` and
``Float64Index`` are supported as index to dataframes and series.
Constructing them in Bodo functions, passing them to Bodo functions (unboxing),
and returning them from Bodo functions (boxing) are also supported.

* :func:`pandas.RangeIndex`

   * ``start``, ``stop`` and ``step`` arguments are supported.

* :func:`pandas.Int64Index`
* :func:`pandas.UInt64Index`
* :func:`pandas.Float64Index`

  * ``data``, ``copy`` and ``name`` arguments are supported.
    ``data`` can be a list or array.


DatetimeIndex
~~~~~~~~~~~~~

``DatetimeIndex`` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

* :func:`pandas.DatetimeIndex`

   * Only ``data`` argument is supported, and can be array-like
     of ``datetime64['ns']``, ``int64`` or strings.
     Strings should be in ISO 8601 format,
     YYYY-MM-DDT[HH[:MM[:SS[.mmm[uuu]]]]][+HH:MM] (e.g. '2017-09-27').

Date fields of DatetimeIndex are supported:

* :attr:`DatetimeIndex.year`
* :attr:`DatetimeIndex.month`
* :attr:`DatetimeIndex.day`
* :attr:`DatetimeIndex.hour`
* :attr:`DatetimeIndex.minute`
* :attr:`DatetimeIndex.second`
* :attr:`DatetimeIndex.microsecond`
* :attr:`DatetimeIndex.nanosecond`
* :attr:`DatetimeIndex.date`

The min/max methods are supported without optional arguments
(``NaT`` output for empty or all ``NaT`` input not supported yet):

* :meth:`DatetimeIndex.min`
* :meth:`DatetimeIndex.max`

Returning underlying data array:

* :attr:`DatetimeIndex.values`


Subtraction of ``Timestamp`` from ``DatetimeIndex`` and vice versa
is supported.

Comparison operators ``==``, ``!=``, ``>=``, ``>``, ``<=``, ``<`` between
``DatetimeIndex`` and a string containing datetime in ISO 8601 format
are supported.


TimedeltaIndex
~~~~~~~~~~~~~~

``TimedeltaIndex`` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

* :func:`pandas.TimedeltaIndex`

   * Only ``data`` argument is supported, and can be array-like
     of ``timedelta64['ns']`` or ``int64``.

Time fields of TimedeltaIndex are supported:

* :attr:`TimedeltaIndex.days`
* :attr:`TimedeltaIndex.second`
* :attr:`TimedeltaIndex.microsecond`
* :attr:`TimedeltaIndex.nanosecond`

PeriodIndex
~~~~~~~~~~~~~

``PeriodIndex`` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.


Timestamp
~~~~~~~~~

* :attr:`Timestamp.day`
* :attr:`Timestamp.hour`
* :attr:`Timestamp.microsecond`
* :attr:`Timestamp.month`
* :attr:`Timestamp.nanosecond`
* :attr:`Timestamp.second`
* :attr:`Timestamp.year`

* :meth:`Timestamp.date`

Window
~~~~~~

* :meth:`Rolling.count`
* :meth:`Rolling.sum`
* :meth:`Rolling.mean`
* :meth:`Rolling.median`
* :meth:`Rolling.var`
* :meth:`Rolling.std`
* :meth:`Rolling.min`
* :meth:`Rolling.max`
* :meth:`Rolling.corr`
* :meth:`Rolling.cov`
* :meth:`Rolling.apply`


GroupBy
~~~~~~~


* :meth:`GroupBy.apply`
* :meth:`GroupBy.count`
* :meth:`GroupBy.max`
* :meth:`GroupBy.mean`
* :meth:`GroupBy.median`
* :meth:`GroupBy.min`
* :meth:`GroupBy.prod`
* :meth:`GroupBy.std`
* :meth:`GroupBy.sum`
* :meth:`GroupBy.var`
