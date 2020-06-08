.. _pandas:

Supported Pandas Operations
---------------------------

Below is the reference list of the Pandas data types and operations that Bodo supports.
Overall, Bodo currently supports 252 of 1263 Pandas APIs (excluding 645 date offset APIs).
Optional arguments are not supported unless if specified.


Comparing to `PySpark DataFrames <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame>`_
(as of version 2.4.5), some equivalent form for 47 of 53 applicable methods
are supported (`colRegex`, `cube`, `freqItems`, `rollup` and `sampleBy` not supported yet).
Comparing to `PySpark SQL functions <https://spark.apache.org/docs/latest/api/python/pyspark.sql#module-pyspark.sql.functions>`_,
some equivalent form for 128 of 205 applicable methods are supported (others will be supported in the future).

.. _pandas-dtype:

Data Types
~~~~~~~~~~

Bodo supports the following
data types as values in Pandas Dataframe and Series data structures.
This represents all `Pandas data types <https://pandas.pydata.org/pandas-docs/stable/reference/arrays.html>`_
except `TZ-aware datetime`, `Period`, `Interval`, and `Sparse` (will be supported in the future).
Comparing to Spark (as of version 2.4.5), equivalent of all
`Spark data types <https://spark.apache.org/docs/latest/sql-reference.html#data-types>`_
are supported except `MapType` and `StructType` (will be supported in the future).


* Numpy booleans: `np.bool_`.
* Numpy integer data types: `np.int8`, `np.int16`, `np.int32`, `np.int64`,
  `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`.
* Numpy floating point data types: `np.float32`, `np.float64`.
* Numpy datetime data types: `np.dtype("datetime64[ns]")` and `np.dtype("timedelta[ns]")`.
  The resolution has to be `ns` currently, which covers most practical use cases.
* Numpy complex data types: `np.complex64` and `np.complex128`.
* Strings (including nulls).
* `datetime.date` values (including nulls).
* Pandas `nullable integers <https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html>`_.
* Pandas `nullable booleans <https://pandas.pydata.org/pandas-docs/stable/user_guide/boolean.html>`_.
* Pandas `Categoricals <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_
  (limited support currently, category values have to be known at compilation).
* Lists of integer, float, and string values.
* `decimal.Decimal` values (including nulls). The decimal
  values are stored as fixed-precision
  `Apache Arrow Decimal128 <https://arrow.apache.org/docs/cpp/api/utilities.html#classarrow_1_1_decimal128>`_
  format, which is also similar to
  `PySpark decimals <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html>`_.
  The decimal type has a `precision` (the maximum total number of digits)
  and a `scale` (the number of digits on the right of dot) attribute, specifying how
  the stored data is interpreted. For example, the (4, 2) case can store from -999.99 to 999.99.
  The precision can be up to 38, and the scale must be less or equal to precision.
  Arbitrary-precision Python `decimal.Decimal` values are converted with precision of 38 and scale of 18.


.. _pandas-f-in:

Input/Output
~~~~~~~~~~~~

Also see :ref:`S3` and :ref:`HDFS` configuration requirements and more on :ref:`file_io`.

* :func:`pandas.read_csv`

  * :ref:`example usage and more system specific instructions <csv-section>`
  * ``filepath_or_buffer`` should be a string and is required. It could be pointing to a single CSV file, or a directory containing multiple partitioned CSV files (must have ``csv`` file extension inside directory).  
  * Arguments ``sep``, ``delimiter``, ``header``, ``names``,
    ``index_col``, ``usecols``, ``dtype``, ``skiprows``, and ``parse_dates`` are supported.
  * Either ``names`` and ``dtype`` arguments should be provided to enable type inference,
    or ``filepath_or_buffer`` should be a constant string for Bodo to infer types by looking at the file at compile time.
  * ``names``, ``usecols``, ``parse_dates`` should be constant lists.
  * ``dtype`` should be a constant dictionary of strings and types.
  * When a CSV file is read in parallel (distributed mode) and each process reads only a portion of the file, reading columns that contain line breaks is not supported.

* :func:`pandas.read_excel`

  * output dataframe cannot be parallelized automatically yet.
  * only arguments ``io``, ``sheet_name``, ``header``, ``names``, ``comment``, ``dtype``, ``skiprows``, ``parse_dates`` are supported.
  * ``io`` should be a string and is required.
  * Either ``names`` and ``dtype`` arguments should be provided to enable type inference,
    or ``io`` should be a constant string for Bodo to infer types by looking at the file at
    compile time.
  * ``sheet_name``, ``header``, ``comment``, and ``skiprows`` should be constant if provided.
  * ``names`` and ``parse_dates`` should be constant lists if provided.
  * ``dtype`` should be a constant dictionary of strings and types if provided.

* :func:`pandas.read_sql`

  * :ref:`example usage and more system specific instructions <sql-section>`
  * Argument ``sql`` is supported but only as a string form. SQLalchemy `Selectable` is not supported. There is no restriction on the form of the sql request.
  * Argument ``con`` is supported but only as a string form. SQLalchemy `connectable` is not supported.
  * Argument ``index_col`` is supported.
  * Arguments ``chunksize``, ``column``, ``coerce_float``, ``params`` are not supported.

* :func:`pandas.read_parquet`

  * :ref:`example usage and more system specific instructions <parquet-section>`
  * Arguments ``path`` and ``columns`` are supported. ``columns``
    should be a constant list of strings.
  * If ``path`` is constant, Bodo finds the schema from file at compilation time.
    Otherwise, schema should be provided. For example::

      @bodo.jit(locals={'df':{'A': bodo.float64[:],
                              'B': bodo.string_array_type}})
      def impl(f):
        df = pd.read_parquet(f)
        return df

* :func:`pandas.read_json`

  * :ref:`Example usage and more system specific instructions <json-section>`
  * Only supports reading `JSON Lines text file format <http://jsonlines.org/>`_ (``pd.read_json(filepath_or_buffer, orient='records', lines=True)``) and regular multi-line JSON file(``pd.read_json(filepath_or_buffer, orient='records', lines=False)``).
  * Argument ``filepath_or_buffer`` is supported: it can point to a single JSON file, or a directory containing multiple partitioned JSON files. When reading a directory, the JSON files inside the directory must be `JSON Lines text file format <http://jsonlines.org/>`_ with ``json`` file extension.
  * Argument ``orient = 'records'`` is used as default, instead of Pandas' default ``'columns'`` for dataframes. ``'records'`` is the only supported value for ``orient``.
  * Argument ``typ`` is supported. ``'frame'`` is the only supported value for ``typ``.
  * ``dtype`` argument should be provided to enable type inference, or ``filepath_or_buffer`` should be a constant string for Bodo to infer types by looking at the file at compile time (not supported for multi-line JSON files)
  * Arguments ``convert_dates``, ``precise_float``, ``lines`` are supported.

* :func:`pandas.to_sql`

  * :ref:`example usage and more system specific instructions <sql-section>`
  * Argument ``con`` is supported but only as a string form. SQLalchemy `connectable` is not supported.
  * Argument ``name``, ``schema``, ``if_exists``, ``index``, ``index_label``, ``dtype``, ``method`` are supported.
  * Argument ``chunksize`` is not supported.

General functions
~~~~~~~~~~~~~~~~~

Data manipulations:

* :func:`pandas.crosstab`

  * Annotation of pivot values is required.
    For example, `@bodo.jit(pivots={'pt': ['small', 'large']})` declares
    the output table `pt` will have columns called `small` and `large`.

* :func:`pandas.merge`

  * Arguments ``left``, ``right`` should be dataframes.
  * ``how``, ``on``, ``left_on``, ``right_on``, ``left_index``,
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

* :func:`pandas.to_numeric` Input can be a Series or array.
  Output type is float64 by default.
  Unlike Pandas, Bodo does not dynamically determine output type,
  and does not downcast to the smallest numerical type.
  `downcast` parameter should be used for type annotation of output.
  The `errors` argument is not supported currently (errors will be coerced by default).


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
* :meth:`pandas.Series.round` (`decimals` argument supported)
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
* :meth:`pandas.Series.all` only default arguments supported
* :meth:`pandas.Series.any` only default arguments supported
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
* :meth:`pandas.Series.isin`
  `values` argument supports both distributed array/Series and replicated list/array/Series
* :meth:`pandas.Series.rename` (only set a new name using a string value)
* :meth:`pandas.Series.reset_index` only default arguments supported.
  Also, requires Index name to be known at compilation time.
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
* :meth:`pandas.DataFrame.assign` function arguments not supported yet.
* :meth:`pandas.DataFrame.join` only dataframes. The output dataframe is not sorted by default for better parallel performance (Pandas may preserve key order depending on `how`).
  One can use explicit sort if needed.
* :meth:`pandas.DataFrame.merge` only dataframes. `how`, `on`, `left_on`,
  `right_on`, `left_index`, and `right_index` are supported but
  should be constant values.



Time series-related:

* :meth:`pandas.DataFrame.shift` (supports numeric types and
  only the `periods` argument supported)

.. _pandas-f-out:

Serialization / IO / conversion:

Also see :ref:`S3` and :ref:`HDFS` configuration requirements and more on :ref:`file_io`.

* :meth:`pandas.DataFrame.to_parquet`
* :meth:`pandas.DataFrame.to_csv`
* :meth:`pandas.DataFrame.to_json`

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
* :meth:`pandas.core.groupby.GroupBy.first`
* :meth:`pandas.core.groupby.GroupBy.last`
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
