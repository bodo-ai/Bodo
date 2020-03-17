.. _pandas:

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
  * When a CSV file is read in parallel (distributed mode) and each process reads only a portion of the file, reading columns that contain line breaks is not supported.

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
  * The output dataframe is not sorted by default for better parallel performance (Pandas may preserve key order depending on `how`). 
    One can use explicit sort if needed.

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


Top-level dealing with datetime like:


* :func:`pandas.date_range`

  * ``start``, ``end``, ``periods``, ``freq``, ``name`` and ``closed``
    arguments are supported. This function is not parallelized yet.


Series
~~~~~~

Bodo provides extensive Series support.
However, operations between Series (+, -, /, *, **) do not
implicitly align values based on their
associated index values yet.


* :class:`pandas.Series`

  * Arguments ``data``, ``index``, and ``name`` are supported.
    ``data`` is required and can be a list, array, Series or Index.
    If ``data`` is Series and ``index`` is provided, implicit alignment is
    not performed yet.


Attributes:

* :attr:`pandas.Series.index`
* :meth:`pandas.Series.values`
* :meth:`pandas.Series.dtype` (object data types such as dtype of
  string series not supported yet)
* :meth:`pandas.Series.shape`
* :meth:`pandas.Series.ndim`
* :meth:`pandas.Series.size`
* :meth:`pandas.Series.T`
* :meth:`pandas.Series.hasnans`
* :meth:`pandas.Series.empty`
* :meth:`pandas.Series.dtypes`
* :meth:`pandas.Series.name`


Methods:

Conversion:

* :meth:`pandas.Series.astype` (only ``dtype`` argument,
  can be a Numpy numeric dtype or ``str``)
* :meth:`pandas.Series.copy` (including ``deep`` argument)
* :meth:`pandas.Series.to_list`
* :meth:`pandas.Series.to_numpy`


Indexing, iteration:

Location based indexing using `[]`, `iat`, and `iloc` is supported.
Changing values of existing string Series using these operators
is not supported yet.

* :meth:`pandas.Series.iat`
* :meth:`pandas.Series.iloc`
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
* :meth:`pandas.Series.combine`
* :meth:`pandas.Series.lt`
* :meth:`pandas.Series.gt`
* :meth:`pandas.Series.le`
* :meth:`pandas.Series.ge`
* :meth:`pandas.Series.ne`

Function application, GroupBy & Window:

* :meth:`pandas.Series.apply` (only the `func` argument)
* :meth:`pandas.Series.map` (only the `arg` argument, which should be a function)
* :meth:`pandas.Series.rolling` (`window` and `center` arguments supported)


Computations / Descriptive Stats:

Statistical functions below are supported without optional arguments
unless support is explicitly mentioned.

* :meth:`pandas.Series.abs`
* :meth:`pandas.Series.corr`
* :meth:`pandas.Series.count`
* :meth:`pandas.Series.cov`
* :meth:`pandas.Series.cumsum`
* :meth:`pandas.Series.cumprod`
* :meth:`pandas.Series.cummin`
* :meth:`pandas.Series.cummax`
* :meth:`pandas.Series.describe` currently returns a string instead of Series object.
* :meth:`pandas.Series.max`
* :meth:`pandas.Series.mean`
* :meth:`pandas.Series.median`
* :meth:`pandas.Series.min`
* :meth:`pandas.Series.nlargest` (non-numerics not supported yet)
* :meth:`pandas.Series.nsmallest` (non-numerics not supported yet)
* :meth:`pandas.Series.pct_change`(supports numeric types and
  only the `periods` argument supported)
* :meth:`pandas.Series.prod`
* :meth:`pandas.Series.quantile`
* :meth:`pandas.Series.std`
* :meth:`pandas.Series.sum`
* :meth:`pandas.Series.var`
* :meth:`pandas.Series.unique`
* :meth:`pandas.Series.nunique`
* :meth:`pandas.Series.value_counts`


Reindexing / Selection / Label manipulation:


* :meth:`pandas.Series.head` (`n` argument is supported)
* :meth:`pandas.Series.idxmax`
* :meth:`pandas.Series.idxmin`
* :meth:`pandas.Series.rename` (only set a new name using a string value)
* :meth:`pandas.Series.tail` (`n` argument is supported)
* :meth:`pandas.Series.take`

Missing data handling:

* :meth:`pandas.Series.isna`
* :meth:`pandas.Series.notna`
* :meth:`pandas.Series.dropna`
* :meth:`pandas.Series.fillna`

Reshaping, sorting:

* :meth:`pandas.Series.argsort`
* :meth:`pandas.Series.sort_values`
* :meth:`pandas.Series.append` `ignore_index` is supported.
  setting name for output Series not supported yet)

Time series-related:

* :meth:`pandas.Series.shift` (supports numeric types and
  only the `periods` argument supported)

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

String handling:

* :meth:`pandas.Series.str.capitalize`
* :meth:`pandas.Series.str.center`
* :meth:`pandas.Series.str.contains` regex argument supported.
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
* :meth:`pandas.Series.str.lstrip`
* :meth:`pandas.Series.str.pad`
* :meth:`pandas.Series.str.replace` regex argument supported.
* :meth:`pandas.Series.str.rfind`
* :meth:`pandas.Series.str.rjust`
* :meth:`pandas.Series.str.rstrip`
* :meth:`pandas.Series.str.slice`
* :meth:`pandas.Series.str.split`
* :meth:`pandas.Series.str.startswith`
* :meth:`pandas.Series.str.strip`
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


DataFrame
~~~~~~~~~

Bodo provides extensive DataFrame support documented below.


* :class:`pandas.DataFrame`

  ``data`` argument can be a constant dictionary or 2d Numpy array.
  Other arguments are also supported.

Attributes and underlying data:


* :attr:`pandas.DataFrame.index` (can access but not set new index yet)
* :attr:`pandas.DataFrame.columns`  (can access but not set new columns yet)
* :attr:`pandas.DataFrame.values` (only for numeric dataframes)
* :meth:`pandas.DataFrame.to_numpy` (only for numeric dataframes)
* :attr:`pandas.DataFrame.ndim`
* :attr:`pandas.DataFrame.size`
* :attr:`pandas.DataFrame.shape`
* :attr:`pandas.DataFrame.empty`

Conversion:

* :meth:`pandas.DataFrame.astype` (only accepts a single data type
  of Numpy dtypes or `str`)
* :meth:`pandas.DataFrame.copy` (including `deep` flag)
* :meth:`pandas.DataFrame.isna`
* :meth:`pandas.DataFrame.notna`


Indexing, iteration:

* :meth:`pandas.DataFrame.head` (including `n` argument)
* :meth:`pandas.DataFrame.iat`
* :meth:`pandas.DataFrame.iloc`
  Read support for all indexers except reading a single row using an
  interger, slicing across columns, or using a `callable` object.
  Label-based indexing is not supported yet.
* :meth:`pandas.DataFrame.tail` (including `n` argument)
* :meth:`pandas.DataFrame.isin` (`values` can be a dataframe with matching index
  or a list or a set)
* :meth:`pandas.DataFrame.query` (`expr` can be a constant string or an argument
  to the jit function)

Function application, GroupBy & Window:

* :meth:`pandas.DataFrame.apply`
* :meth:`pandas.DataFrame.groupby` `by` should be a constant column label
  or column labels.
  `sort=False` is set by default. `as_index` argument is supported but
  `MultiIndex` is not supported yet (will just drop output `MultiIndex`).
* :meth:`pandas.DataFrame.rolling` `window` argument should be integer or a time
  offset as a constant string. `center` and `on` arguments are also supported.

Computations / Descriptive Stats:

* :meth:`pandas.DataFrame.abs`
* :meth:`pandas.DataFrame.corr` (`min_periods` argument supported)
* :meth:`pandas.DataFrame.count`
* :meth:`pandas.DataFrame.cov` (`min_periods` argument supported)
* :meth:`pandas.DataFrame.cumprod`
* :meth:`pandas.DataFrame.cumsum`
* :meth:`pandas.DataFrame.cummin`
* :meth:`pandas.DataFrame.cummax`
* :meth:`pandas.DataFrame.describe`
* :meth:`pandas.DataFrame.max`
* :meth:`pandas.DataFrame.mean`
* :meth:`pandas.DataFrame.median`
* :meth:`pandas.DataFrame.min`
* :meth:`pandas.DataFrame.pct_change`
* :meth:`pandas.DataFrame.prod`
* :meth:`pandas.DataFrame.quantile`
* :meth:`pandas.DataFrame.sum`
* :meth:`pandas.DataFrame.std`
* :meth:`pandas.DataFrame.var`
* :meth:`pandas.DataFrame.nunique` (`dropna` argument not supported yet. The behavior is slightly different from `.nunique` implementation in pandas)


Reindexing / Selection / Label manipulation:

* :meth:`pandas.DataFrame.drop` (only dropping columns supported,
  either using `columns` argument or setting `axis=1`)
* :meth:`pandas.DataFrame.drop_duplicates`
* :meth:`pandas.DataFrame.duplicated`
* :meth:`pandas.DataFrame.head` (including `n` argument)
* :meth:`pandas.DataFrame.idxmax`
* :meth:`pandas.DataFrame.idxmin`
* :meth:`pandas.DataFrame.rename` (only `columns` argument with a constant dictionary)
* :meth:`pandas.DataFrame.reset_index` (only `drop=True` supported)
* :meth:`pandas.DataFrame.set_index` `keys` can only be a column label
  (a constant string).
* :meth:`pandas.DataFrame.tail` (including `n` argument)
* :meth:`pandas.DataFrame.take`

Missing data handling:

* :meth:`pandas.DataFrame.dropna`
* :meth:`pandas.DataFrame.fillna`

Reshaping, sorting, transposing:

* :meth:`pandas.DataFrame.pivot_table`

  * Arguments ``values``, ``index``, ``columns`` and ``aggfunc`` are
    supported.
  * Annotation of pivot values is required.
    For example, `@bodo.jit(pivots={'pt': ['small', 'large']})` declares
    the output pivot table `pt` will have columns called `small` and `large`.

* :meth:`pandas.DataFrame.sort_values` ``by`` argument should be constant string or
  constant list of strings. ``ascending`` and ``na_position`` arguments are supported.
* :meth:`pandas.DataFrame.drop_duplicates` is supported.
* :meth:`pandas.DataFrame.sort_index` `ascending` argument is supported.

Combining / joining / merging:

* :meth:`pandas.DataFrame.append` appending a dataframe or list of dataframes
  supported. `ignore_index=True` is necessary and set by default.
* :meth:`pandas.DataFrame.join` only dataframes. The output dataframe is not sorted by default for better parallel performance (Pandas may preserve key order depending on `how`).
  One can use explicit sort if needed.
* :meth:`pandas.DataFrame.merge` only dataframes. `how`, `on`, `left_on`,
  `right_on`, `left_index`, and `right_index` are supported but
  should be constant values.



Time series-related:

* :meth:`pandas.DataFrame.shift` (supports numeric types and
  only the `periods` argument supported)


Numeric Index
~~~~~~~~~~~~~

Numeric index objects ``RangeIndex``, ``Int64Index``, ``UInt64Index`` and
``Float64Index`` are supported as index to dataframes and series.
Constructing them in Bodo functions, passing them to Bodo functions (unboxing),
and returning them from Bodo functions (boxing) are also supported.

* :class:`pandas.RangeIndex`

  * ``start``, ``stop`` and ``step`` arguments are supported.

* :class:`pandas.Int64Index`
* :class:`pandas.UInt64Index`
* :class:`pandas.Float64Index`

  * ``data``, ``copy`` and ``name`` arguments are supported.
    ``data`` can be a list or array.


DatetimeIndex
~~~~~~~~~~~~~

``DatetimeIndex`` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

* :class:`pandas.DatetimeIndex`

  * Only ``data`` argument is supported, and can be array-like
    of ``datetime64['ns']``, ``int64`` or strings.

Date fields of DatetimeIndex are supported:

* :meth:`pandas.DatetimeIndex.year`
* :meth:`pandas.DatetimeIndex.month`
* :meth:`pandas.DatetimeIndex.day`
* :meth:`pandas.DatetimeIndex.hour`
* :meth:`pandas.DatetimeIndex.minute`
* :meth:`pandas.DatetimeIndex.second`
* :meth:`pandas.DatetimeIndex.microsecond`
* :meth:`pandas.DatetimeIndex.nanosecond`
* :meth:`pandas.DatetimeIndex.date`

The min/max methods are supported without optional arguments
(``NaT`` output for empty or all ``NaT`` input not supported yet):

* :meth:`pandas.DatetimeIndex.min`
* :meth:`pandas.DatetimeIndex.max`

Returning underlying data array:

* :attr:`pandas.DatetimeIndex.values`


Subtraction of ``Timestamp`` from ``DatetimeIndex`` and vice versa
is supported.

Comparison operators ``==``, ``!=``, ``>=``, ``>``, ``<=``, ``<`` between
``DatetimeIndex`` and a string of datetime
are supported.


TimedeltaIndex
~~~~~~~~~~~~~~

``TimedeltaIndex`` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

* :class:`pandas.TimedeltaIndex`

  * Only ``data`` argument is supported, and can be array-like
    of ``timedelta64['ns']`` or ``int64``.

Time fields of TimedeltaIndex are supported:

* :meth:`pandas.TimedeltaIndex.days`
* :meth:`pandas.TimedeltaIndex.seconds`
* :meth:`pandas.TimedeltaIndex.microseconds`
* :meth:`pandas.TimedeltaIndex.nanoseconds`


PeriodIndex
~~~~~~~~~~~~~

``PeriodIndex`` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.


Timestamp
~~~~~~~~~

Timestamp functionality is documented in `pandas.Timestamp <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html>`_.

* :attr:`pandas.Timestamp.day`
* :attr:`pandas.Timestamp.hour`
* :attr:`pandas.Timestamp.microsecond`
* :attr:`pandas.Timestamp.month`
* :attr:`pandas.Timestamp.nanosecond`
* :attr:`pandas.Timestamp.second`
* :attr:`pandas.Timestamp.year`
* :meth:`pandas.Timestamp.date`


Window
~~~~~~

Rolling functionality is documented in `pandas.DataFrame.rolling <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html>`_.

* :meth:`pandas.core.window.rolling.Rolling.count`
* :meth:`pandas.core.window.rolling.Rolling.sum`
* :meth:`pandas.core.window.rolling.Rolling.mean`
* :meth:`pandas.core.window.rolling.Rolling.median`
* :meth:`pandas.core.window.rolling.Rolling.var`
* :meth:`pandas.core.window.rolling.Rolling.std`
* :meth:`pandas.core.window.rolling.Rolling.min`
* :meth:`pandas.core.window.rolling.Rolling.max`
* :meth:`pandas.core.window.rolling.Rolling.corr`
* :meth:`pandas.core.window.rolling.Rolling.cov`
* :meth:`pandas.core.window.rolling.Rolling.apply`


GroupBy
~~~~~~~

The operations are documented on `pandas.DataFrame.groupby <https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html>`_.

* :meth:`pandas.core.groupby.GroupBy.agg` `arg` should be a function, and the compiler should be
  able to simplify it to a single parallel loop and analyze it.
  For example, arithmetic expressions on input Series are supported.
  A list of functions is also supported if one output column is selected
  (which avoids MultiIndex).
  For example::

    @bodo.jit
    def f(df):
        def g1(x): return (x<=2).sum()
        def g2(x): return (x>2).sum()
        return df.groupby('A')['B'].agg((g1, g2))

* :meth:`pandas.core.groupby.GroupBy.aggregate` same as `agg`
* :meth:`pandas.core.groupby.GroupBy.count`
* :meth:`pandas.core.groupby.GroupBy.cumsum`
* :meth:`pandas.core.groupby.GroupBy.max`
* :meth:`pandas.core.groupby.GroupBy.mean`
* :meth:`pandas.core.groupby.GroupBy.min`
* :meth:`pandas.core.groupby.GroupBy.prod`
* :meth:`pandas.core.groupby.GroupBy.std`
* :meth:`pandas.core.groupby.GroupBy.sum`
* :meth:`pandas.core.groupby.GroupBy.var`


Integer NA issue in Pandas
~~~~~~~~~~~~~~~~~~~~~~~~~~

DataFrame and Series objects with integer data need special care
due to `integer NA issues in Pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions>`_.
By default, Pandas dynamically converts integer columns to
floating point when missing values (NAs) are needed
(which can result in loss of precision).
This is because Pandas uses the NaN floating point value as NA,
and Numpy does not support NaN values for integers.
Bodo does not perform this conversion unless enough information is
available at compilation time.

Pandas introduced a new `nullable integer data type <https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html#integer-na>`_
that can solve this issue, which is also supported by Bodo.
For example, this code reads column `A` into a nullable integer array
(the capital "I" denotes nullable integer type)::

  @bodo.jit
  def example(fname):
    dtype = {'A': 'Int64', 'B': 'float64'}
    df = pd.read_csv(fname,
        names=dtype.keys(),
        dtype=dtype,
    )
    ...

Bodo can use nullable integer arrays when reading Parquet files if
the `bodo.io.parquet_pio.use_nullable_int_arr` flag is set by the user.
For example::

  bodo.io.parquet_pio.use_nullable_int_arr = True
  @bodo.jit
  def example(fname):
    df = pd.read_parquet(fname)
    ...
