.. _pandas:

Supported Pandas Operations
---------------------------

Below is a reference list of the Pandas data types and operations that Bodo supports.
This list will expand regularly as we add support for more APIs.
Optional arguments are not supported unless if specified.

.. Overall, Bodo currently supports 252 of 1263 Pandas APIs (excluding 645 date offset APIs).

.. Comparing to `PySpark DataFrames <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame>`_
.. (as of version 2.4.5), some equivalent form for 47 of 53 applicable methods
.. are supported (`colRegex`, `cube`, `freqItems`, `rollup` and `sampleBy` not supported yet).
.. Comparing to `PySpark SQL functions <https://spark.apache.org/docs/latest/api/python/pyspark.sql#module-pyspark.sql.functions>`_,
.. some equivalent form for 128 of 205 applicable methods are supported (others will be supported in the future).

.. _pandas-dtype:

Data Types
~~~~~~~~~~

Bodo supports the following
data types as values in Pandas Dataframe and Series data structures.
This represents all `Pandas data types <https://pandas.pydata.org/pandas-docs/stable/reference/arrays.html>`_
except `TZ-aware datetime`, `Period`, `Interval`, and `Sparse` (which will be supported in the future).
Comparing to Spark, equivalent of all
`Spark data types <http://spark.apache.org/docs/latest/sql-ref-datatypes.html>`_
are supported.


* Numpy booleans: `np.bool_`.
* Numpy integer data types: `np.int8`, `np.int16`, `np.int32`, `np.int64`,
  `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`.
* Numpy floating point data types: `np.float32`, `np.float64`.
* Numpy datetime data types: `np.dtype("datetime64[ns]")` and `np.dtype("timedelta[ns]")`.
  The resolution has to be `ns` currently, which covers most practical use cases.
* Numpy complex data types: `np.complex64` and `np.complex128`.
* Strings (including nulls).
* `datetime.date` values (including nulls).
* `datetime.timedelta` values (including nulls).
* Pandas `nullable integers <https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html>`_.
* Pandas `nullable booleans <https://pandas.pydata.org/pandas-docs/stable/user_guide/boolean.html>`_.
* Pandas `Categoricals <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_.
* Lists of other data types.
* Tuples of other data types.
* Structs of other data types.
* Maps of other data types (each map is a set of key-value pairs). All keys should have the same type
  to ensure type stability. All values should have the same type as well.
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


In addition, it may be desirable to specify type annotations in some cases (e.g. :ref:`file I/O array input types <input-array-types>`).
Typically these types are array types and they all can be accessed directly from the `bodo` module.
The following table can be used to select the necessary Bodo Type based upon the desired Python, Numpy, or Pandas type.

.. list-table::
  :header-rows: 1

  * - Bodo Type Name
    - Equivalent Python, Numpy, or Pandas type
  * - ``bodo.bool_[:]``, ``bodo.int8[:]``, ..., ``bodo.int64[:]``, ``bodo.uint8[:]``, ..., ``bodo.uint64[:]``, ``bodo.float32[:]``, ``bodo.float64[:]``
    - One-dimensional Numpy array of the given type. A full list of supported Numpy types can be found `here <https://numba.readthedocs.io/en/stable/reference/types.html#numbers>`_.
      A multidimensional can be specified by adding additional colons (e.g. ``bodo.int32[:, :, :]`` for a three-dimensional array).
  * - ``bodo.string_array_type``
    - Array of nullable strings
  * - ``bodo.IntegerArrayType(integer_type)``
    - | Array of Pandas nullable integers of the given integer type
      | e.g. ``bodo.IntegerArrayType(bodo.int64)``
  * - ``bodo.boolean_array``
    - Array of Pandas nullable booleans
  * - ``bodo.datetime64ns[:]``
    - Array of Numpy datetime64 values
  * - ``bodo.timedelta64ns[:]``
    - Array of Numpy timedelta64 values
  * - ``bodo.datetime_date_array_type``
    - Array of datetime.date types
  * - ``bodo.datetime_timedelta_array_type``
    - Array of datetime.timedelta types
  * - ``bodo.DecimalArrayType(precision, scale)``
    - | Array of Apache Arrow Decimal128 values with the given precision and scale
      | e.g. ``bodo.DecimalArrayType(38, 18)``
  * - ``bodo.binary_array_type``
    - Array of nullable bytes values
  * - ``bodo.StructArrayType(data_types, field_names)``
    - | Array of a user defined struct with the given tuple of data types and field names
      | e.g. ``bodo.StructArrayType((bodo.int32[:], bodo.datetime64ns[:]), ("a", "b"))``
  * - ``bodo.TupleArrayType(data_types)``
    - | Array of a user defined tuple with the given tuple of data types
      | e.g. ``bodo.TupleArrayType((bodo.int32[:], bodo.datetime64ns[:]))``
  * - ``bodo.MapArrayType(key_arr_type, value_arr_type)``
    - | Array of Python dictionaries with the given key and value array types.
      | e.g. ``bodo.MapArrayType(bodo.uint16[:], bodo.string_array_type)``
  * - ``bodo.PDCategoricalDtype(cat_tuple, cat_elem_type, is_ordered_cat)``
    - | Pandas categorical type with the possible categories, each category's type, and if the categories are ordered.
      | e.g. ``bodo.PDCategoricalDtype(("A", "B", "AA"), bodo.string_type, True)``
  * - ``bodo.CategoricalArrayType(categorical_type)``
    - | Array of Pandas categorical values.
      | e.g. ``bodo.CategoricalArrayType(bodo.PDCategoricalDtype(("A", "B", "AA"), bodo.string_type, True))``
  * - ``bodo.DatetimeIndexType(name_type)``
    - | Index of datetime64 values with a given name type.
      | e.g. ``bodo.DatetimeIndexType(bodo.string_type)``
  * - ``bodo.NumericIndexType(data_type, name_type)``
    - | Index of pd.Int64, pd.Uint64, or Float64 objects,
        based upon the given data_type and name type.
      | e.g. ``bodo.NumericIndexType(bodo.float64, bodo.string_type)``
  * - ``bodo.PeriodIndexType(freq, name_type)``
    - | pd.PeriodIndex with a given frequency and name type.
      | e.g. ``bodo.PeriodIndexType('A', bodo.string_type)``
  * - ``bodo.RangeIndexType(name_type)``
    - | RangeIndex with a given name type.
      | e.g. ``bodo.RangeIndexType(bodo.string_type)``
  * - ``bodo.StringIndexType(name_type)``
    - | Index of strings with a given name type.
      | e.g. ``bodo.StringIndexType(bodo.string_type)``
  * - ``bodo.TimedeltaIndexType(name_type)``
    - | Index of timedelta64 values with a given name type.
      | e.g. ``bodo.TimedeltaIndexType(bodo.string_type)``
  * - ``bodo.SeriesType(dtype=data_type, index=index_type, name_typ=name_type)``
    - | Series with a given data type, index type, and name type.
      | e.g. ``bodo.SeriesType(bodo.float32, bodo.DatetimeIndexType(bodo.string_type), bodo.string_type)``
  * - ``bodo.DataFrameType(data_types_tuple, index_type, column_names)``
    - | DataFrame with a tuple of data types, an index type, and the names of the columns.
      | e.g. ``bodo.DataFrameType((bodo.int64[::1], bodo.float64[::1]), bodo.RangeIndexType(bodo.none), ("A", "B"))``






.. _pandas-f-in:

Input/Output
~~~~~~~~~~~~

See more in :ref:`file_io`, such as :ref:`S3` and :ref:`HDFS` configuration requirements.

* :func:`pandas.read_csv`

  * :ref:`example usage and more system specific instructions <csv-section>`
  * ``filepath_or_buffer`` should be a string and is required. It could be pointing to a single CSV file, or a directory containing multiple partitioned CSV files (must have ``csv`` file extension inside directory).
  * Arguments ``sep``, ``delimiter``, ``header``, ``names``,
    ``index_col``, ``usecols``, ``dtype``, ``skiprows``, and ``parse_dates`` are supported.
  * Either ``names`` and ``dtype`` arguments should be provided to enable type inference,
    or ``filepath_or_buffer`` should be inferrable as a constant string for Bodo to infer types by looking at the file at compile time.
  * ``names``, ``usecols``, ``parse_dates`` should be constant lists.
  * ``dtype`` should be a constant dictionary of strings and types.
  * When a CSV file is read in parallel (distributed mode) and each process reads only a portion of the file, reading columns that contain line breaks is not supported.

* :func:`pandas.read_excel`

  * output dataframe cannot be parallelized automatically yet.
  * only arguments ``io``, ``sheet_name``, ``header``, ``names``, ``comment``, ``dtype``, ``skiprows``, ``parse_dates`` are supported.
  * ``io`` should be a string and is required.
  * Either ``names`` and ``dtype`` arguments should be provided to enable type inference,
    or ``io`` should be inferrable as a constant string for Bodo to infer types by looking at the file at
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
    should be a constant list of strings if provided.
  * Argument ``anon`` of ``storage_options`` is supported for S3 filepaths.
  * If ``path`` can be inferred as a constant (e.g. it is a function argument),
    Bodo finds the schema from file at compilation time.
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
  * ``dtype`` argument should be provided to enable type inference, or ``filepath_or_buffer`` should be inferrable as a constant string for Bodo
    to infer types by looking at the file at compile time (not supported for multi-line JSON files)
  * Arguments ``convert_dates``, ``precise_float``, ``lines`` are supported.

* :func:`pandas.DataFrame.to_sql`

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
    ``right_index``, and ``indicator`` are supported but should be constant values.
  * The output dataframe is not sorted by default for better parallel performance (Pandas may preserve key order depending on `how`).
    One can use explicit sort if needed.

* :func:`pandas.merge_asof` (similar arguments to `merge`)

* :func:`pandas.concat`
  Input list or tuple of dataframes or series is supported. `axis` and `ignore_index` are also supported.
  Bodo currently concatenates local data chunks for distributed datasets, which does not preserve global order of concatenated objects in output.

* :func:`pandas.get_dummies`
  Input must be a categorical array with categories that are known at compile time (for type stability).


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



Top-level dealing with datetime and timedelta like:


* :func:`pandas.date_range`

  * ``start``, ``end``, ``periods``, ``freq``, ``name`` and ``closed``
    arguments are supported. This function is not parallelized yet.

* :func:`pandas.to_datetime`

  * All arguments are supported.

* :func:`pandas.to_timedelta`

  * ``arg_a`` and ``unit`` arguments are supported.


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
* :attr:`pandas.Series.ndim`
* :attr:`pandas.Series.size`
* :attr:`pandas.Series.T`
* :attr:`pandas.Series.hasnans`
* :attr:`pandas.Series.empty`
* :attr:`pandas.Series.dtypes`
* :attr:`pandas.Series.name`
* :attr:`pandas.Series.nbytes`
* :attr:`pandas.Series.is_monotonic`
* :attr:`pandas.Series.is_monotonic_increasing`
* :attr:`pandas.Series.is_monotonic_decreasing`


Methods:

Conversion:

* :meth:`pandas.Series.astype` (only ``dtype`` argument)
* :meth:`pandas.Series.copy` (including ``deep`` argument)
* :meth:`pandas.Series.to_list`
* :meth:`pandas.Series.tolist`
* :meth:`pandas.Series.to_numpy`


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
* :meth:`pandas.Series.combine`
* :meth:`pandas.Series.round` (`decimals` argument supported)
* :meth:`pandas.Series.lt`
* :meth:`pandas.Series.gt`
* :meth:`pandas.Series.le`
* :meth:`pandas.Series.ge`
* :meth:`pandas.Series.ne`
* :meth:`pandas.Series.dot`

Function application, GroupBy & Window:

* :meth:`pandas.Series.apply` (`convert_dtype` not supported yet)
* :meth:`pandas.Series.map` (only the `arg` argument, which should be a function or dictionary)
* :meth:`pandas.Series.groupby` (pass array to `by` argument, or level=0 with regular Index,
  `sort=False` and `observed=True` are set by default)
* :meth:`pandas.Series.rolling` (`window`, `min_periods` and `center` arguments supported)


Computations / Descriptive Stats:

Statistical functions below are supported without optional arguments
unless support is explicitly mentioned.

* :meth:`pandas.Series.abs`
* :meth:`pandas.Series.all`
* :meth:`pandas.Series.any`
* :meth:`pandas.Series.corr`
* :meth:`pandas.Series.count`
* :meth:`pandas.Series.cov` (supports ddof)
* :meth:`pandas.Series.cumsum`
* :meth:`pandas.Series.cumprod`
* :meth:`pandas.Series.cummin`
* :meth:`pandas.Series.cummax`
* :meth:`pandas.Series.describe` (supports numeric types. Assumes `datetime_is_numeric=True`.)
* :meth:`pandas.Series.diff` (Implemented for Numpy Array data types. Supports `periods` argument.)
* :meth:`pandas.Series.max`
* :meth:`pandas.Series.mean`
* :meth:`pandas.Series.autocorr` (supports `lag` argument)
* :meth:`pandas.Series.median` (supports `skipna` argument)
* :meth:`pandas.Series.min`
* :meth:`pandas.Series.nlargest` (non-numerics not supported yet)
* :meth:`pandas.Series.nsmallest` (non-numerics not supported yet)
* :meth:`pandas.Series.pct_change` (supports numeric types and
  only the `periods` argument supported)
* :meth:`pandas.Series.pipe` `func` should be a function (not tuple)
* :meth:`pandas.Series.prod`
* :meth:`pandas.Series.product`
* :meth:`pandas.Series.quantile`
* :meth:`pandas.Series.std` (support `skipna` and `ddof` arguments)
* :meth:`pandas.Series.var` (support `skipna` and `ddof` arguments)
* :meth:`pandas.Series.sem` (support `skipna` and `ddof` arguments)
* :meth:`pandas.Series.sum`
* :meth:`pandas.Series.mad` argument `skipna` supported
* :meth:`pandas.Series.kurt` argument `skipna` supported
* :meth:`pandas.Series.kurtosis` argument `skipna` supported
* :meth:`pandas.Series.skew` argument `skipna` supported
* :meth:`pandas.Series.unique` the output is assumed to be "small" relative to input and is replicated.
  Use Series.drop_duplicates() if the output should remain distributed.
* :meth:`pandas.Series.nunique` all optional arguments are supported
* :meth:`pandas.Series.value_counts` all optional arguments except `dropna` are supported.
* :meth:`pandas.Series.between`
* :meth:`pandas.Series.memory_usage` argument `index` supported


Reindexing / Selection / Label manipulation:


* :meth:`pandas.Series.head` (`n` argument is supported)
* :meth:`pandas.Series.idxmax`
* :meth:`pandas.Series.idxmin`
* :meth:`pandas.Series.isin`
  `values` argument supports both distributed array/Series and replicated list/array/Series
* :meth:`pandas.Series.rename` (only set a new name using a string value)
* :meth:`pandas.Series.reset_index` For MultiIndex case, only dropping all levels supported.
  Requires Index name to be known at compilation time if `drop=False`.
* :meth:`pandas.Series.tail` (`n` argument is supported)
* :meth:`pandas.Series.take`
* :meth:`pandas.Series.equals` (series and `other` should contain scalar values in each row)
* :meth:`pandas.Series.where` (`cond` and `other` arguments supported for 1d numpy data arrays. Categorical data supported for scalar 'other'.)
* :meth:`pandas.Series.mask` (`cond` and `other` arguments supported for 1d numpy data arrays. Categorical data supported for scalar 'other'.)

Missing data handling:

* :meth:`pandas.Series.isna`
* :meth:`pandas.Series.isnull`
* :meth:`pandas.Series.notna`
* :meth:`pandas.Series.dropna`
* :meth:`pandas.Series.fillna`
* :meth:`pandas.Series.replace`

Reshaping, sorting:

* :meth:`pandas.Series.argsort`
* :meth:`pandas.Series.sort_index` `ascending` and `na_position` arguments are supported
* :meth:`pandas.Series.sort_values` `ascending` and `na_position` arguments are supported
* :meth:`pandas.Series.append` `ignore_index` is supported.
  setting name for output Series not supported yet)
* :meth:`pandas.Series.explode`
* :meth:`pandas.Series.repeat`

Time series-related:

* :meth:`pandas.Series.shift` (supports numeric, boolean, datetime.date, and string types, and
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
* :attr:`pandas.Series.dt.quarter`
* :attr:`pandas.Series.dt.dayofweek`
* :attr:`pandas.Series.dt.dayofyear`
* :attr:`pandas.Series.dt.daysinmonth`
* :attr:`pandas.Series.dt.days_in_month`
* :attr:`pandas.Series.dt.is_month_start`
* :attr:`pandas.Series.dt.is_month_end`
* :attr:`pandas.Series.dt.is_quarter_start`
* :attr:`pandas.Series.dt.is_quarter_end`
* :attr:`pandas.Series.dt.is_year_start`
* :attr:`pandas.Series.dt.is_year_end`
* :attr:`pandas.Series.dt.week`
* :attr:`pandas.Series.dt.weekday`
* :attr:`pandas.Series.dt.weekofyear`
* :meth:`pandas.Series.dt.ceil`
* :meth:`pandas.Series.dt.day_name` (``locale`` not supported)
* :meth:`pandas.Series.dt.floor`
* :meth:`pandas.Series.dt.month_name` (``locale`` not supported)
* :meth:`pandas.Series.dt.normalize`
* :meth:`pandas.Series.dt.round`
* :meth:`pandas.Series.dt.strftime`

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
* :meth:`pandas.Series.str.lstrip`
* :meth:`pandas.Series.str.pad`
* :meth:`pandas.Series.str.repeat`
* :meth:`pandas.Series.str.replace` `regex` argument supported.
* :meth:`pandas.Series.str.rfind`
* :meth:`pandas.Series.str.rjust`
* :meth:`pandas.Series.str.rstrip`
* :meth:`pandas.Series.str.slice`
* :meth:`pandas.Series.str.slice_replace`
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

Serialization / Conversion

* :meth:`pandas.Series.to_dict` is not parallelized since dictionaries are not parallelized
* :meth:`pandas.Series.to_frame` Series name should be a known constant or a constant 'name' should be provided

DataFrame
~~~~~~~~~

Bodo provides extensive DataFrame support documented below.


* :class:`pandas.DataFrame`

  ``data`` argument can be a constant dictionary or 2D Numpy array.
  Other arguments are also supported.

Attributes and underlying data:


* :attr:`pandas.DataFrame.columns`
* :attr:`pandas.DataFrame.dtypes`
* :attr:`pandas.DataFrame.empty`
* :attr:`pandas.DataFrame.index`
* :attr:`pandas.DataFrame.ndim`
* :meth:`pandas.DataFrame.select_dtypes` (only supports constant strings or types as arguments)
* :attr:`pandas.DataFrame.filter` (only supports the column axis)
* :attr:`pandas.DataFrame.shape`
* :attr:`pandas.DataFrame.size`
* :meth:`pandas.DataFrame.to_numpy` (only for numeric dataframes)
* :attr:`pandas.DataFrame.values` (only for numeric dataframes)

Conversion:

* :meth:`pandas.DataFrame.astype` (only `dtype` argument)
* :meth:`pandas.DataFrame.copy` (including `deep` flag)
* :meth:`pandas.DataFrame.isna`
* :meth:`pandas.DataFrame.isnull`
* :meth:`pandas.DataFrame.notna`
* :meth:`pandas.DataFrame.notnull`
* :meth:`pandas.DataFrame.info`

Indexing, iteration:

* :meth:`pandas.DataFrame.head` (including `n` argument)
* :meth:`pandas.DataFrame.iat`
* :meth:`pandas.DataFrame.iloc`
* :meth:`pandas.DataFrame.insert` (`loc`, `column`, and `allow_duplicates` should be constant inputs)
* :meth:`pandas.DataFrame.isin` (`values` can be a dataframe with matching index
  or a list or a set)
* :meth:`pandas.DataFrame.itertuples`
  Read support for all indexers except reading a single row using an
  interger, slicing across columns, or using a `callable` object.
  Label-based indexing is not supported yet.
* :meth:`pandas.DataFrame.query` (`expr` can be a constant string or an argument
  to the jit function)
* :meth:`pandas.DataFrame.tail` (including `n` argument)

Function application, GroupBy & Window:

* :meth:`pandas.DataFrame.apply`
  Supports extra `_bodo_inline` boolean argument to manually control bodo's inlining behavior.
  Inlining user-defined functions (UDFs) can potentially improve performance at the expense of
  extra compilation time. Bodo uses heuristics to make a decision automatically if `_bodo_inline` is not provided.
* :meth:`pandas.DataFrame.groupby` `by` should be a constant column label
  or column labels.
  `sort=False` and `observed=True` are set by default. `as_index`  and `dropna` arguments are supported.
* :meth:`pandas.DataFrame.rolling` `window` argument should be integer or a time
  offset (as a constant string, pd.Timedelta, or datetime.timedelta).
  `min_periods`, `center` and `on` arguments are also supported.
  `on` should be a constant column name.

Computations / Descriptive Stats:

* :meth:`pandas.DataFrame.abs`
* :meth:`pandas.DataFrame.corr` (`min_periods` argument supported)
* :meth:`pandas.DataFrame.count`
* :meth:`pandas.DataFrame.cov` (`min_periods` argument supported)
* :meth:`pandas.DataFrame.cummax`
* :meth:`pandas.DataFrame.cummin`
* :meth:`pandas.DataFrame.cumprod`
* :meth:`pandas.DataFrame.cumsum`
* :meth:`pandas.DataFrame.describe` (supports numeric types. Assumes `datetime_is_numeric=True`.)
* :meth:`pandas.DataFrame.diff` (Implemented for Numpy Array data types. Supports `periods` argument.)
* :meth:`pandas.DataFrame.max`
* :meth:`pandas.DataFrame.mean`
* :meth:`pandas.DataFrame.median`
* :meth:`pandas.DataFrame.min`
* :meth:`pandas.DataFrame.nunique` all optional arguments are supported
* :meth:`pandas.DataFrame.pct_change`
* :meth:`pandas.DataFrame.pipe` `func` should be a function (not tuple)
* :meth:`pandas.DataFrame.prod`
* :meth:`pandas.DataFrame.product`
* :meth:`pandas.DataFrame.quantile`
* :meth:`pandas.DataFrame.std`
* :meth:`pandas.DataFrame.sum`
* :meth:`pandas.DataFrame.var`
* :meth:`pandas.DataFrame.memory_usage` argument `index` supported


Reindexing / Selection / Label manipulation:

* :meth:`pandas.DataFrame.drop`

  * Only dropping columns supported, either using `columns` argument or setting `axis=1`
  * `inplace` supported with a constant boolean value
* :meth:`pandas.DataFrame.drop_duplicates`
* :meth:`pandas.DataFrame.duplicated`
* :meth:`pandas.DataFrame.head` (including `n` argument)
* :meth:`pandas.DataFrame.idxmax`
* :meth:`pandas.DataFrame.idxmin`
* :meth:`pandas.DataFrame.rename` (can only rename columns with a constant dictionary, either through `columns` or `mapper` and `axis=1`)
* :meth:`pandas.DataFrame.reset_index` (only dropping all levels supported. `drop` and `inplace` also supported)
* :meth:`pandas.DataFrame.set_index` (`keys` must be a constant string column label)
* :meth:`pandas.DataFrame.tail` (including `n` argument)
* :meth:`pandas.DataFrame.take`

Missing data handling:

* :meth:`pandas.DataFrame.dropna`

  * Arguments ``how``, ``thresh`` and ``subset`` are supported.

* :meth:`pandas.DataFrame.fillna`

  * Arguments ``value`` and ``inplace`` are supported.

* :meth:`pandas.DataFrame.replace`

Reshaping, sorting, transposing:

* :meth:`pandas.DataFrame.pivot_table`

  * Arguments ``values``, ``index``, ``columns`` and ``aggfunc`` are
    supported.
  * Annotation of pivot values is required.
    For example, `@bodo.jit(pivots={'pt': ['small', 'large']})` declares
    the output pivot table `pt` will have columns called `small` and `large`.

* :meth:`pandas.DataFrame.sample` is supported except for the arguments ``random_state``, ``weights`` and ``axis``.
* :meth:`pandas.DataFrame.sort_index` `ascending` and `na_position` arguments supported.
* :meth:`pandas.DataFrame.sort_values` ``by`` argument should be constant string or
  constant list of strings. ``ascending`` and ``na_position`` arguments are supported.
* :meth:`pandas.DataFrame.to_string` (not distributed since output is a string)

Combining / joining / merging:

* :meth:`pandas.DataFrame.append` appending a dataframe or list of dataframes
  supported. `ignore_index` is supported.
* :meth:`pandas.DataFrame.assign` function arguments not supported yet.
* :meth:`pandas.DataFrame.join` only dataframes. The output dataframe is not sorted by default for better parallel performance (Pandas may preserve key order depending on `how`).
  One can use explicit sort if needed.
* :meth:`pandas.DataFrame.merge` only dataframes. `how`, `on`, `left_on`,
  `right_on`, `left_index`, `right_index`, and `indicator` are supported but
  should be constant values.



Time series-related:

* :meth:`pandas.DataFrame.shift` (supports numeric, boolean, datetime.date and string types, and
  only the `periods` argument supported)

.. _pandas-f-out:

Serialization / IO / conversion:

Also see :ref:`S3` and :ref:`HDFS` configuration requirements and more on :ref:`file_io`.

* :meth:`pandas.DataFrame.to_csv`
* :meth:`pandas.DataFrame.to_json`
* :meth:`pandas.DataFrame.to_parquet`
* :meth:`pandas.DataFrame.to_sql`

Index objects
~~~~~~~~~~~~~

Index
*****

Properties

* :attr:`pandas.Index.name`
* :attr:`pandas.Index.shape`
* :attr:`pandas.Index.values`
  Returns the underlying data array
* :attr:`pandas.Index.nbytes`

Modifying and computations:

* :meth:`pandas.Index.copy`
* :meth:`pandas.Index.take`
* :meth:`pandas.Index.min`
* :meth:`pandas.Index.max`

|  The min/max methods are supported for DatetimeIndex. They are supported without optional arguments
|  (``NaT`` output for empty or all ``NaT`` input not supported yet):


Missing values:

* :meth:`pandas.Index.isna`

Conversion:

* :meth:`pandas.Index.map`



Numeric Index
*************

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
*************

``DatetimeIndex`` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

* :class:`pandas.DatetimeIndex`

  * Only ``data`` argument is supported, and can be array-like
    of ``datetime64['ns']``, ``int64`` or strings.

Date fields of DatetimeIndex are supported:

* :attr:`pandas.DatetimeIndex.year`
* :attr:`pandas.DatetimeIndex.month`
* :attr:`pandas.DatetimeIndex.day`
* :attr:`pandas.DatetimeIndex.hour`
* :attr:`pandas.DatetimeIndex.minute`
* :attr:`pandas.DatetimeIndex.second`
* :attr:`pandas.DatetimeIndex.microsecond`
* :attr:`pandas.DatetimeIndex.nanosecond`
* :attr:`pandas.DatetimeIndex.date`
* :attr:`pandas.DatetimeIndex.is_leap_year`
* :attr:`pandas.DatetimeIndex.is_month_start`
* :attr:`pandas.DatetimeIndex.is_month_end`
* :attr:`pandas.DatetimeIndex.is_quarter_start`
* :attr:`pandas.DatetimeIndex.is_quarter_end`
* :attr:`pandas.DatetimeIndex.is_year_start`
* :attr:`pandas.DatetimeIndex.is_year_end`
* :attr:`pandas.DatetimeIndex.week`
* :attr:`pandas.DatetimeIndex.weekday`
* :attr:`pandas.DatetimeIndex.weekofyear`
* :attr:`pandas.DatetimeIndex.quarter`




Subtraction of ``Timestamp`` from ``DatetimeIndex`` and vice versa
is supported.

Comparison operators ``==``, ``!=``, ``>=``, ``>``, ``<=``, ``<`` between
``DatetimeIndex`` and a string of datetime
are supported.


TimedeltaIndex
**************

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

Min and Max operators are supported:

* :meth:`pandas.TimedeltaIndex.min`
* :meth:`pandas.TimedeltaIndex.max`

PeriodIndex
***********

``PeriodIndex`` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.


MultiIndex
**********

* :meth:`pandas.MultiIndex.from_product` (*iterables* and *names* supported as tuples, no parallel support yet)


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
* :attr:`pandas.Timestamp.components`
* :attr:`pandas.Timestamp.dayofyear`
* :attr:`pandas.Timestamp.dayofweek`
* :attr:`pandas.Timestamp.days_in_month`
* :attr:`pandas.Timestamp.daysinmonth`
* :attr:`pandas.Timestamp.is_leap_year`
* :attr:`pandas.Timestamp.is_month_start`
* :attr:`pandas.Timestamp.is_month_end`
* :attr:`pandas.Timestamp.is_quarter_start`
* :attr:`pandas.Timestamp.is_quarter_end`
* :attr:`pandas.Timestamp.is_year_start`
* :attr:`pandas.Timestamp.is_year_end`
* :attr:`pandas.Timestamp.quarter`
* :attr:`pandas.Timestamp.week`
* :attr:`pandas.Timestamp.weekofyear`
* :meth:`pandas.Timestamp.ceil`
* :meth:`pandas.Timestamp.date`
* :meth:`pandas.Timestamp.day_name` (``locale`` not supported)
* :meth:`pandas.Timestamp.floor`
* :meth:`pandas.Timestamp.isocalendar`
* :meth:`pandas.Timestamp.isoformat`
* :meth:`pandas.Timestamp.month_name` (``locale`` not supported)
* :meth:`pandas.Timestamp.normalize`
* :meth:`pandas.Timestamp.round`
* :meth:`pandas.Timestamp.strftime`
* :meth:`pandas.Timestamp.toordinal`
* :meth:`pandas.Timestamp.weekday`


Timedelta
~~~~~~~~~
Timedelta functionality is documented in `pandas.Timedelta <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html>`_.

* :class:`pandas.Timedelta`

  * The unit argument is not supported and all Timedeltas
    are represented in nanosecond precision.

Datetime related fields are supported:

* :attr:`pandas.Timedelta.components`
* :attr:`pandas.Timedelta.days`
* :attr:`pandas.Timedelta.delta`
* :attr:`pandas.Timedelta.microseconds`
* :attr:`pandas.Timedelta.nanoseconds`
* :attr:`pandas.Timedelta.seconds`
* :attr:`pandas.Timedelta.value`
* :meth:`pandas.Timedelta.ceil`
* :meth:`pandas.Timedelta.floor`
* :meth:`pandas.Timedelta.round`
* :meth:`pandas.Timedelta.to_numpy`
* :meth:`pandas.Timedelta.to_pytimedelta`
* :meth:`pandas.Timedelta.to_timedelta64`
* :meth:`pandas.Timedelta.total_seconds`



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
* :meth:`pandas.core.window.rolling.Rolling.apply` (`raw` argument supported)


GroupBy
~~~~~~~

The operations are documented on `pandas.DataFrame.groupby <https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html>`_.

* :meth:`pandas.core.groupby.GroupBy.apply` (`func` should return a DataFrame or Series)
* :meth:`pandas.core.groupby.GroupBy.agg` `func` should be a function or constant dictionary
  of input/function mappings.
  Passing a list of functions is also supported if only one output column is selected.
  Alternatively, outputs can be specified using keyword arguments and `pd.NamedAgg()`.
* :meth:`pandas.core.groupby.DataFrameGroupBy.aggregate` same as `agg`
* :meth:`pandas.core.groupby.GroupBy.count`
* :meth:`pandas.core.groupby.GroupBy.cumsum`
* :meth:`pandas.core.groupby.GroupBy.first`
* :meth:`pandas.core.groupby.GroupBy.last`
* :meth:`pandas.core.groupby.GroupBy.max`
* :meth:`pandas.core.groupby.GroupBy.mean`
* :meth:`pandas.core.groupby.GroupBy.min`
* :meth:`pandas.core.groupby.GroupBy.pipe` `func` should be a function (not tuple)
* :meth:`pandas.core.groupby.GroupBy.prod`
* :meth:`pandas.core.groupby.GroupBy.rolling`
* :meth:`pandas.core.groupby.GroupBy.std`
* :meth:`pandas.core.groupby.GroupBy.sum`
* :meth:`pandas.core.groupby.GroupBy.var`
* :meth:`pandas.core.groupby.DataFrameGroupBy.idxmin`
* :meth:`pandas.core.groupby.DataFrameGroupBy.idxmax`
* :meth:`pandas.core.groupby.DataFrameGroupBy.nunique` all optional arguments are supported
* :meth:`pandas.core.groupby.GroupBy.median`
* :meth:`pandas.core.groupby.GroupBy.shift`
* :meth:`pandas.core.groupby.GroupBy.size`
* :meth:`pandas.core.groupby.SeriesGroupBy.value_counts`
* :meth:`pandas.core.groupby.DataFrameGroupBy.transform` (only `'count'`, `'min'`, `'max'`, `'mean'`, `'std'`, and `'sum'` operations are supported)


Offsets
~~~~~~~

Bodo supports a subset of the offset types in ``pandas.tseries.offsets``:

* :func:`pandas.tseries.offsets.DateOffset`
* :func:`pandas.tseries.offsets.MonthBegin`
* :func:`pandas.tseries.offsets.MonthEnd`
* :func:`pandas.tseries.offsets.Week`

The currently supported operations are the constructor
and addition and subtraction with a scalar `datetime.date`, `datetime.datetime`
or `pandas.Timestamp`. These can also be mapped across Series or DataFrame of
dates using UDFs.


.. _integer-na-issue-pandas:

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

User-Defined Functions (UDFs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User-defined functions (UDFs) can be applied to dataframes with ``DataFrame.apply()`` and to
series with ``Series.apply()`` or ``Series.map()``. Bodo offers support for UDFs without the
significant runtime penalty generally incurred in Pandas.

It is recommended to pass additional variables to UDFs explicitly, instead of directly using
values in the main function. The latter results in the "captured" variables case, which is
often error-prone and may result in compilation errors. Therefore, arguments should be passed
directly to either ``Series.apply()`` or ``DataFrame.apply()``.
The Bodo compiler transforms the code to pass main function values as arguments to `apply()` automatically if possible.

For example, consider a UDF that appends a variable suffix to each string
in a Series of strings. The proper way to write this function through ``Series.apply()`` is::

    @bodo.jit
    def add_suffix(S, suffix):
        return S.apply(lambda x, suf: x + suf, args=(suffix,))

Alternatively, arguments can be passed as named arguments like::

    @bodo.jit
    def add_suffix(S, suffix):
        return S.apply(lambda x, suf: x + suf, suf=suffix)

The same process can be applied in the Dataframe case using ``DataFrame.apply()``.


Type Inference for Object Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pandas stores some data types (e.g. strings) as object arrays which are untyped.
Therefore, Bodo needs to infer the actual data type of object arrays
when dataframes or series values are passed
to JIT functions from regular Python.
Bodo uses the first non-null value of the array to determine the type,
and throws a warning if the array is empty or all nulls:

.. code-block:: none

  BodoWarning: Empty object array passed to Bodo, which causes ambiguity in typing. This can cause errors in parallel execution.

In this case, Bodo assumes the array is a string array which is the most common.
However, this can cause errors if a distributed dataset is passed to Bodo, and some other
processor has non-string data.
This corner case can usually be avoided by load balancing
the data across processors to avoid empty arrays.
