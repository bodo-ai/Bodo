.. _pandas:

Pandas Operations
-----------------

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


In addition, it may be desirable to specify type annotations in some cases (e.g. :ref:`file I/O array input types <non-constant-filepaths>`).
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
  * - ``bodo.BinaryIndexType(name_type)``
    - | Index of binary values with a given name type.
      | e.g. ``bodo.BinaryIndexType(bodo.string_type)``
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
    ``index_col``, ``usecols``, ``dtype``, ``nrows``, ``skiprows``, ``chunksize``, ``parse_dates``, and ``low_memory`` are supported.
  * Either ``names`` and ``dtype`` arguments should be provided to enable type inference,
    or ``filepath_or_buffer`` should be inferrable as a constant string. This is required so bodo can infer the types at compile time, see `compile time constants <https://docs.bodo.ai/latest/source/programming_with_bodo/require_constants.html>`
  * ``names``, ``usecols``, ``parse_dates`` should be constant lists.
  * ``dtype`` should be a constant dictionary of strings and types.
  * ``skiprows`` must be an integer or list of integers and if it is not a constant, ``names`` must be provided to enable type inference.
  * ``chunksize`` is supported for uncompressed files only.
  * ``low_memory`` internally process file in chunks while parsing. In Bodo this is set to `False` by default.
  * When set to `True`, Bodo parses file in chunks but like Pandas the entire file is read into a single DataFrame regardless.
  * If you want to load data in chunks, use the ``chunksize`` argument.
  * When a CSV file is read in parallel (distributed mode) and each process reads only a portion of the file, reading columns that contain line breaks is not supported.

* :func:`pandas.read_excel`

  * output dataframe cannot be parallelized automatically yet.
  * only arguments ``io``, ``sheet_name``, ``header``, ``names``, ``comment``, ``dtype``, ``skiprows``, ``parse_dates`` are supported.
  * ``io`` should be a string and is required.
  * Either ``names`` and ``dtype`` arguments should be provided to enable type inference,
    or ``io`` should be inferrable as a constant string. This is required so bodo can infer the types at compile time, see `compile time constants <https://docs.bodo.ai/latest/source/programming_with_bodo/require_constants.html>`
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
    Otherwise, schema should be provided using the `numba syntax <https://numba.pydata.org/numba-doc/latest/reference/types.html>`. For example::

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
  * ``filepath_or_buffer`` must be inferrable as a constant string. This is required so bodo can infer the types at compile time, see `compile time constants <https://docs.bodo.ai/latest/source/programming_with_bodo/require_constants.html>`.
  * Arguments ``convert_dates``, ``precise_float``, ``lines`` are supported.


General functions
~~~~~~~~~~~~~~~~~

Data manipulations:

* :func:`pandas.crosstab` ``(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)``

  Supported Arguments:

      * index:

        - SeriesType

      * columns:

        - SeriesType

  Important Notes:

    Annotation of pivot values is required. For example,
    ``@bodo.jit(pivots={'pt': ['small', 'large']})`` declares
    the output table `pt` will have columns called ``small`` and ``large``.

  Example Usage::

     >>> @bodo.jit(pivots={"pt": ["small", "large"]})
     ... def f(df):
     ...   pt = pd.crosstab(df.A, df.C)
     ...   return pt

     >>> list_A = ["foo", "foo", "bar", "bar", "bar", "bar"]
     >>> list_C = ["small", "small", "large", "small", "small", "middle"]
     >>> df = pd.DataFrame({"A": list_A, "C": list_C})
     >>> f(df)

           small  large
    index
    foo        2      0
    bar        2      1

* :func:`pandas.cut` ``(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates="raise", ordered=True)``

  Supported Arguments:

    * x:

      - Series or Array like

    * bins:

      - Integer or Array like

    * include_lowest (default=False):

      - Boolean

  Example Usage::

     >>> @bodo.jit
     ... def f(S):
     ...   bins = 4
     ...   include_lowest = True
     ...   return pd.cut(S, bins, include_lowest=include_lowest)

     >>> S = pd.Series(
     ...    [-2, 1, 3, 4, 5, 11, 15, 20, 22],
     ...    ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"],
     ...    name="ABC",
     ... )
     >>> f(S)

    a1    (-2.025, 4.0]
    a2    (-2.025, 4.0]
    a3    (-2.025, 4.0]
    a4    (-2.025, 4.0]
    a5      (4.0, 10.0]
    a6     (10.0, 16.0]
    a7     (10.0, 16.0]
    a8     (16.0, 22.0]
    a9     (16.0, 22.0]
    Name: ABC, dtype: category
    Categories (4, interval[float64, right]): [(-2.025, 4.0] < (4.0, 10.0] < (10.0, 16.0] < (16.0, 22.0]]


* :func:`pandas.qcut` ``(x, q, labels=None, retbins=False, precision=3, duplicates="raise")``

  Supported Arguments:
    * x:

      - Series or Array like

    * q:

      - Integer or Array like of floats

  Example Usage::

     >>> @bodo.jit
     ... def f(S):
     ...   q = 4
     ...   return pd.qcut(S, q)

     >>> S = pd.Series(
     ...      [-2, 1, 3, 4, 5, 11, 15, 20, 22],
     ...      ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"],
     ...      name="ABC",
     ... )
     >>> f(S)

     a1    (-2.001, 3.0]
     a2    (-2.001, 3.0]
     a3    (-2.001, 3.0]
     a4       (3.0, 5.0]
     a5       (3.0, 5.0]
     a6      (5.0, 15.0]
     a7      (5.0, 15.0]
     a8     (15.0, 22.0]
     a9     (15.0, 22.0]
     Name: ABC, dtype: category
     Categories (4, interval[float64, right]): [(-2.001, 3.0] < (3.0, 5.0] < (5.0, 15.0] < (15.0, 22.0]]


.. _pd_merge_fn:

* :func:`pandas.merge` ``(left, right, how="inner", on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=("_x", "_y"), copy=True, indicator=False, validate=None, _bodo_na_equal=True)``

  Supported Arguments:

    * left:

      - DataFrame

    * right:

      - DataFrame

    * how (default='inner'):

      - **Must be constant at Compile Time**
      - String
      - Must be one of "inner", "outer", "left", "right"

    * on (default=None):

      - **Must be constant at Compile Time**
      - Column Name, List of Column Names, or General Merge Condition
        String (see important notes).

    * left_on (default=None):

      - **Must be constant at Compile Time**
      - Column Name or List of Column Names

    * right_on (default=None):

      - **Must be constant at Compile Time**
      - Column Name or List of Column Names

    * left_index (default=False):

      - **Must be constant at Compile Time**
      - Boolean

    * right_index (default=False):

      - **Must be constant at Compile Time**
      - Boolean

    * suffixes (default=('_x', '_y')):

      - **Must be constant at Compile Time**
      - Tuple of Strings

    * indicator (default=False):

      - **Must be constant at Compile Time**
      - Boolean

    * _bodo_na_equal (default=True):

      - **Must be constant at Compile Time**
      - Boolean
      - This argument is unique to Bodo and not available in Pandas.
        If False, Bodo won't consider NA/nan keys as equal, which differs
        from Pandas.



  Important Notes:

    * Output Ordering:

      The output dataframe is not sorted by default for better parallel performance
      (Pandas may preserve key order depending on `how`).
      One can use explicit sort if needed.

    * General Merge Conditions:

      Within Pandas, the merge criteria supported by `pd.merge` are limited to equality between 1
      or more pairs of keys. For some use cases, this is not sufficient and more generalized
      support is necessary. For example, with these limitations, a ``left outer join`` where
      ``df1.A == df2.B & df2.C < df1.A`` cannot be efficiently computed.

      Bodo supports these use cases by allowing users to pass general merge conditions to ``pd.merge``.
      We plan to contribute this feature to Pandas to ensure full compatibility of Bodo and Pandas code.

      General merge conditions are performed by providing the condition as a string via the `on` argument. Columns in the left table
      are referred to by `left.`{column name}`` and columns in the right table are referred to by `right.`{column name}``.

      To execute the example above, a user can call this example.

        .. code:: ipython3

            >>> @bodo.jit
            ... def general_merge(df1, df2):
            ...   return df1.merge(df2, on="left.`A` == right.`B` & right.`C` < left.`A`", how="left")

            >>> df1 = pd.DataFrame({"col": [2, 3, 5, 1, 2, 8], "A": [4, 6, 3, 9, 9, -1]})
            >>> df2 = pd.DataFrame({"B": [1, 2, 9, 3, 2], "C": [1, 7, 2, 6, 5]})
            >>> general_merge(df1, df2)

               col  A     B     C
            0    2  4  <NA>  <NA>
            1    3  6  <NA>  <NA>
            2    5  3  <NA>  <NA>
            3    1  9     9     2
            4    2  9     9     2
            5    8 -1  <NA>  <NA>


      These calls have a few additional requirement:

        * The condition must be constant string.
        * The condition must be of the form ``cond_1 & ... & cond_N`` where at least one ``cond_i``
          is a simple equality. This restriction will be removed in a future release.
        * The columns specified in these conditions are limited to certain column types.
          We currently support `boolean`, `integer`, `float`, `datetime64`, `timedelta64`, `datetime.date`,
          and `string` columns.

  Example Usage::

     >>> @bodo.jit
     ... def f(df1, df2):
     ...   return pd.merge(df1, df2, how="inner", on="key")

     >>> df1 = pd.DataFrame({"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], float)})
     >>> df2 = pd.DataFrame({"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], float)})
     >>> f(df1, df2)

        key    A    B
     0    2  4.0  7.0
     1    2  4.0  5.0
     2    3  6.0  6.0
     3    1  9.0  1.0
     4    2  9.0  7.0
     5    2  9.0  5.0


* :func:`pandas.merge_asof` ``(left, right, on=None, left_on=None, right_on=None, left_index=False, right_index=False, by=None, left_by=None, right_by=None, suffixes=("_x", "_y"), tolerance=None, allow_exact_matches=True, direction="backward")``

  Supported Arguments:

    * left:

      - DataFrame

    * right:

      - DataFrame

    * on (default=None):

      - **Must be constant at Compile Time**
      - Column Name, List of Column Names

    * left_on (default=None):

      - **Must be constant at Compile Time**
      - Column Name or List of Column Names

    * right_on (default=None):

      - **Must be constant at Compile Time**
      - Column Name or List of Column Names

    * left_index (default=False):

      - **Must be constant at Compile Time**
      - Boolean

    * right_index (default=False):

      - **Must be constant at Compile Time**
      - Boolean

    * suffixes (default=('_x', '_y')):

      - **Must be constant at Compile Time**
      - Tuple of Strings

  Example Usage::

     >>> @bodo.jit
     ... def f(df1, df2):
     ...   return pd.merge_asof(df1, df2, on="time")

     >>> df1 = pd.DataFrame(
     ...   {
     ...       "time": pd.DatetimeIndex(["2017-01-03", "2017-01-06", "2017-02-21"]),
     ...       "B": [4, 5, 6],
     ...   }
     ... )
     >>> df2 = pd.DataFrame(
     ...   {
     ...       "time": pd.DatetimeIndex(
     ...           ["2017-01-01", "2017-01-02", "2017-01-04", "2017-02-23", "2017-02-25"]
     ...       ),
     ...       "A": [2, 3, 7, 8, 9],
     ...   }
     ... )
     >>> f(df1, df2)

             time  B  A
     0 2017-01-03  4  3
     1 2017-01-06  5  7
     2 2017-02-21  6  7

* :func:`pandas.concat` ``(objs, axis=0, join="outer", join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True)``

  Supported Arguments:

    * objs:

      - List or Tuple of DataFrames/Series

    * axis (default=0):

      - **Must be constant at Compile Time**
      - Integer with either 0 or 1

    * ignore_index (default=False):

      - **Must be constant at Compile Time**
      - Boolean

  Important Notes:

    Bodo currently concatenates local data chunks for distributed datasets, which does not preserve global order of concatenated objects in output.

  Example Usage::

     >>> @bodo.jit
     ... def f(df1, df2):
     ...     return pd.concat([df1, df2], axis=1)

     >>> df1 = pd.DataFrame({"A": [3, 2, 1, -4, 7]})
     >>> df2 = pd.DataFrame({"B": [3, 25, 1, -4, -24]})
     >>> f(df1, df2)

        A   B
     0  3   3
     1  2  25
     2  1   1
     3 -4  -4
     4  7 -24


* :func:`pandas.get_dummies` ``(data, prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)``

  Supported Arguments:

    * data:

      - Array or Series with Categorical dtypes
      - **Categories must be known at compile time.**

  Example Usage::

     >>> @bodo.jit
     ... def f(S):
     ...     return pd.get_dummies(S)

     >>> S = pd.Series(["CC", "AA", "B", "D", "AA", None, "B", "CC"]).astype("category")
     >>> f(S)

        AA  B  CC  D
     0   0  0   1  0
     1   1  0   0  0
     2   0  1   0  0
     3   0  0   0  1
     4   1  0   0  0
     5   0  0   0  0
     6   0  1   0  0
     7   0  0   1  0

Top-level missing data:

* :func:`pandas.isna` ``(obj)``

  Supported Arguments:

    * obj:

      - DataFrame, Series, Index, Array, or Scalar

  Example Usage::

     >>> @bodo.jit
     ... def f(df):
     ...     return pd.isna(df)

     >>> df = pd.DataFrame(
     ...    {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
     ...    [1.1, -2.1, 7.1, 0.1, 3.1],
     ... )
     >>> f(df)

               A      B
      1.1  False  False
     -2.1   True  False
      7.1  False  False
      0.1  False  False
      3.1  False  False

* :func:`pandas.isnull` ``(obj)``

  Supported Arguments:

    * obj:

      - DataFrame, Series, Index, Array, or Scalar

  Example Usage::

     >>> @bodo.jit
     ... def f(df):
     ...     return pd.isnull(df)

     >>> df = pd.DataFrame(
     ...    {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
     ...    [1.1, -2.1, 7.1, 0.1, 3.1],
     ... )
     >>> f(df)

               A      B
      1.1  False  False
     -2.1   True  False
      7.1  False  False
      0.1  False  False
      3.1  False  False

* :func:`pandas.notna` ``(obj)``

  Supported Arguments:

    * obj:

      - DataFrame, Series, Index, Array, or Scalar

  Example Usage::

     >>> @bodo.jit
     ... def f(df):
     ...     return pd.notna(df)

     >>> df = pd.DataFrame(
     ...    {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
     ...    [1.1, -2.1, 7.1, 0.1, 3.1],
     ... )
     >>> f(df)

               A     B
      1.1   True  True
     -2.1  False  True
      7.1   True  True
      0.1   True  True
      3.1   True  True

* :func:`pandas.notnull` ``(obj)``

  Supported Arguments:

    * obj:

      - DataFrame, Series, Index, Array, or Scalar

  Example Usage::

     >>> @bodo.jit
     ... def f(df):
     ...     return pd.notnull(df)

     >>> df = pd.DataFrame(
     ...    {"A": ["AA", np.nan, "", "D", "GG"], "B": [1, 8, 4, -1, 2]},
     ...    [1.1, -2.1, 7.1, 0.1, 3.1],
     ... )
     >>> f(df)

               A     B
      1.1   True  True
     -2.1  False  True
      7.1   True  True
      0.1   True  True
      3.1   True  True


Top-level conversions:

* :func:`pandas.to_numeric` ``(arg, errors="raise", downcast=None)``

  Supported Arguments:

    * arg:

      - Series or Array

    * downcast (default=None):

      - **Must be constant at Compile Time**
      - String and one of ('integer', 'signed', 'unsigned', 'float')

  Important Notes:

    * Output type is float64 by default

    * Unlike Pandas, Bodo does not dynamically determine output type,
      and does not downcast to the smallest numerical type.

    * `downcast` parameter should be used for type annotation of output.

  Example Usage::

     >>> @bodo.jit
     ... def f(S):
     ...     return pd.to_numeric(S, errors="coerce", downcast="integer")

     >>> S = pd.Series(["1", "3", "12", "4", None, "-555"])
     >>> f(S)

     0       1
     1       3
     2      12
     3       4
     4    <NA>
     5    -555
     dtype: Int64

Top-level dealing with datetime and timedelta like:

* :func:`pandas.to_datetime` ``(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)``

  Supported Arguments:

    * arg:

      - Series, Array or scalar of integers or strings

    * errors (default='raise'):

      - String and one of ('ignore', 'raise', 'coerce')

    * dayfirst (default=False):

      - Boolean

    * yearfirst (default=False):

      - Boolean

    * utc (default=None):

      - Boolean

    * format (default=None):

      - String matching Pandas `strftime/strptime <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_

    * exact (default=True)

      - Boolean

    * unit (default='ns')

      - String

      - Must be a `valid Pandas timedelta unit <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_

    * infer_datetime_format (default=False)

      - Boolean

    * origin (default='unix')

      - Scalar string or timestamp value

    * cache (default=True)

      - Boolean

  Important Notes:

    * The function is not optimized.

    * Bodo doesn't support Timezone-Aware datetime values

  Example Usage::

     >>> @bodo.jit
     ... def f(val):
     ...     return pd.to_datetime(val, format="%Y-%d-%m")

     >>> val = "2016-01-06"
     >>> f(val)

     Timestamp('2016-06-01 00:00:00')


* :func:`pandas.to_timedelta` ``(arg, unit=None, errors='raise')``

  Supported Arguments:

    * arg:

      - Series, Array or scalar of integers or strings

    * unit (default=None):

      - String

      - Must be a `valid Pandas timedelta unit <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_

  Important Notes:

    * Passing string data as ``arg`` is not optimized.

  Example Usage::

     >>> @bodo.jit
     ... def f(S):
     ...     return pd.to_timedelta(S, unit="D")

     >>> S = pd.Series([1.0, 2.2, np.nan, 4.2], [3, 1, 0, -2], name="AA")
     >>> f(val)

      3   1 days 00:00:00
      1   2 days 04:48:00
      0               NaT
     -2   4 days 04:48:00
     Name: AA, dtype: timedelta64[ns]


* :func:`pandas.date_range` ``(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs)``

  Supported Arguments:

    * start (default=None):

      - String or Timestamp

    * end (default=None):

      - String or Timestamp

    * periods (default=None):

      - Integer

    * freq (default=None):

      - String
      - Must be a `valid Pandas frequency <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_

    * name (default=None):

      - String

    * closed (default=None):

      - String and one of ('left', 'right')

  Important Notes:

    * Exactly three of ``start``, ``end``, ``periods``, and ``freq`` must
      be provided.

    * Bodo **Does Not** support ``kwargs``, even for compatibility.

    * This function is not parallelized yet.

  Example Usage::

       >>> @bodo.jit
       ... def f():
       ...     return pd.date_range(start="2018-04-24", end="2018-04-27", periods=3)

       >>> f()

       DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',
                      '2018-04-27 00:00:00'],
                     dtype='datetime64[ns]', freq=None)


* :func:`pandas.timedelta_range` ``(start=None, end=None, periods=None, freq=None, name=None, closed=None)``

  Supported Arguments:

    * start (default=None):

      - String or Timedelta


    * end (default=None):

      - String or Timedelta

    * periods (default=None):

      - Integer

    * freq (default=None):

      - String
      - Must be a `valid Pandas frequency <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_

    * name (default=None):

      - String

    * closed (default=None):

      - String and one of ('left', 'right')

  Important Notes:

    * Exactly three of ``start``, ``end``, ``periods``, and ``freq`` must
      be provided.

    * This function is not parallelized yet.

  Example Usage::

     >>> @bodo.jit
     ... def f():
     ...     return pd.timedelta_range(start="1 day", end="11 days 1 hour", periods=3)

     >>> f()

     TimedeltaIndex(['1 days 00:00:00', '6 days 00:30:00', '11 days 01:00:00'], dtype='timedelta64[ns]', freq=None)


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


DataFrame
~~~~~~~~~

Bodo provides extensive DataFrame support documented below.


* :class:`pandas.DataFrame` ``(data=None, index=None, columns=None, dtype=None, copy=None)``

Supported arguments:
  * data
   - constant key dictionary
   - 2D Numpy array
       * columns argument is required when using a 2D Numpy array
  * index
   - List
   - Tuple
   - pandas index types
   - pandas array types
   - pandas series types
   - numpy array types
  * columns
   - **Must be constant at Compile Time**
   - Constant list of String
   - Constant tuple of String
  * dtype
   - All values supported with dataframe.astype (see below)
  * copy
   - **Must be constant at Compile Time**
   - boolean



Attributes and underlying data:


* :attr:`pandas.DataFrame.columns`

  Example Usage::

      >>> @bodo.jit
      ... def f():
      ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
      ...   return df.columns
      Index(['A', 'B', 'C'], dtype='object')

* :attr:`pandas.DataFrame.dtypes`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
    ...   return df.dtypes
    A              int64
    B             string
    C    timedelta64[ns]
    dtype: object

* :attr:`pandas.DataFrame.empty`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df1 = pd.DataFrame({"A": [1,2,3]})
    ...   df2 = pd.DataFrame()
    ...   return df1.empty, df2.empty
    (False, True)

* :attr:`pandas.DataFrame.index`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3]}, index=["x", "y", "z"])
    ...   return df.index
    Index(['x', 'y', 'z'], dtype='object')

* :attr:`pandas.DataFrame.ndim`

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
    ...   return df.ndim
    2


* :meth:`pandas.DataFrame.select_dtypes` ``(include=None, exclude=None)``

Supported arguments:
  * include
   - **Must be constant at Compile Time**
   - string
   - type
   - List or tuple of string/type
  * exclude
   - **Must be constant at Compile Time**
   - string
   - type
   - List or tuple of string/type



  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df= pd.DataFrame({"A": [1], "B": ["X"], "C": [pd.Timedelta(10, unit="D")], "D": [True], "E": [3.1]})
    ...   out_1 = df_l.select_dtypes(exclude=[np.float64, "bool"])
    ...   out_2 = df_l.select_dtypes(include="int")
    ...   out_3 = df_l.select_dtypes(include=np.bool_, exclude=(np.int64, "timedelta64[ns]"))
    ...   formated_out = "\n".join([out_1.to_string(), out_2.to_string(), out_3.to_string()])
    ...   return formated_out
       A  B       C
    0  1  X 10 days
      A
    0  1
          D
    0  True

* :meth:`pandas.DataFrame.filter` ``(items=None, like=None, regex=None, axis=None)``

Supported arguments:
  * items
   - Constant list of String
  * like
   - Constant string
  * regex
   - Constant String
  * axis (only supports the "column" axis)
   - Constant String
   - Constant Integer



  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]})
    ...   filtered_df_1 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(items = ["A"])
    ...   filtered_df_2 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(like ="hello", axis = "columns")
    ...   filtered_df_3 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(regex="(ab){3}", axis = 1)
    ...   formated_out = "\n".join([filtered_df_1.to_string(), filtered_df_2.to_string(), filtered_df_3.to_string()])
    ...   return formated_out
       A
    0  3
      hello world
    0            2
      ababab
    0       1

* :attr:`pandas.DataFrame.shape`

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [3,4,5]})
    ...   return df.shape
    (3, 2)


* :attr:`pandas.DataFrame.size`

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [3,4,5]})
    ...   return df.size
    6

* :meth:`pandas.DataFrame.to_numpy` ``(dtype=None, copy=False, na_value=NoDefault.no_default)``

Supported Arguments:
  * copy
     - boolean

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
    ...   return df.to_numpy()
    [[1.  3.1]
     [2.  4.2]
     [3.  5.3]]

* :attr:`pandas.DataFrame.values` (only for numeric dataframes)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
    ...   return df.values
    [[1.  3.1]
     [2.  4.2]
     [3.  5.3]]

Conversion:

* :meth:`pandas.DataFrame.astype` ``(dtype, copy=True, errors='raise')``

Supported Arguments:
  * dtype
     - **Must be constant at Compile Time**
     - dict of string column names keys, and Strings/types values
     - String (string must be parsable by np.dtype)
     - Valid type (see types)
     - The following functions: float, int, bool, str


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
    ...   return df.astype({"A": float, "B": "datetime64[ns]"})
         A                             B
    0  1.0 1970-01-01 00:00:00.000000003
    1  2.0 1970-01-01 00:00:00.000000004
    2  3.0 1970-01-01 00:00:00.000000005

* :meth:`pandas.DataFrame.copy` ``(deep=True)``

Supported Arguments:
  * copy
     - boolean


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3]})
    ...   shallow_df = df.copy(deep=False)
    ...   deep_df = df.copy()
    ...   shallow_df["A"][0] = -1
    ...   formated_out = "\n".join([df.to_string(), shallow_df.to_string(), deep_df.to_string()])
    ...   return formated_out
       A
    0  -1
    1  2
    2  3
      A
    0  -1
    1  2
    2  3
      A
    0  1
    1  2
    2  3

* :meth:`pandas.DataFrame.isna` ``()``

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,None,3]})
    ...   return df.isna()
           A
    0  False
    1   True
    2  False

* :meth:`pandas.DataFrame.isnull` ``()``

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,None,3]})
    ...   return df.isnull()
           A
    0  False
    1   True
    2  False


* :meth:`pandas.DataFrame.notna` ``()``

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,None,3]})
    ...   return df.notna()
           A
    0   True
    1  False
    2   True

* :meth:`pandas.DataFrame.notnull` ``()``

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,None,3]})
    ...   return df.notnull()
           A
    0   True
    1  False
    2   True

* :meth:`pandas.DataFrame.info` ``(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None)``

Supported Arguments:
  None

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
    ...   return df.info()
    <class 'DataFrameType'>
    RangeIndexType(none): 3 entries, 0 to 2
    Data columns (total 3 columns):
    #   Column  Non-Null Count  Dtype
    --- ------  --------------  -----
    0  A       3 non-null      int64
    1  B       3 non-null      unicode_type
    2  C       3 non-null      timedelta64[ns]
    dtypes: int64(1), timedelta64[ns](1), unicode_type(1)
    memory usage: 108.0 bytes

.. note::
  The exact output string may vary slightly from Pandas.


Indexing, iteration:

* :meth:`pandas.DataFrame.head` ``(n=5)``

Supported Arguments:
  * head
     - integer

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.DataFrame({"A": np.arange(1000)}).head(3)
       A
    0  0
    1  1
    2  2

* :attr:`pandas.DataFrame.iat`

We only support indexing using iat using a pair of integers. We require that the second int
(the column integer) is a compile time constant


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   df.iat[0, 0] = df.iat[2,2]
    ...   return df
       A  B  C
    0  9  4  7
    1  2  5  8
    2  3  6  9



* :meth:`pandas.DataFrame.iloc`

getitem:
  df.iloc supports single integer indexing (returns row as series) ``df.iloc[0]``

  df.iloc supports single list/array/series of integers/bool ``df.iloc[[0,1,2]]``

  for tuples indexing ``df.iloc[row_idx, col_idx]`` we allow:
    row_idx can be
      int
      list/array/series of integers/bool
      slice

    col_idx can be
      constant int, constant list of integers, or constant slice

    ex:
      ``df.iloc[[0,1,2], :]``

setitem:

  df.iloc only supports scalar setitem

  df.iloc only supports tuple indexing ``df.iloc[row_idx, col_idx]``
    row_idx can be anything supported for series setitem:
      int
      list/array/series of integers/bool
      slice

    col_idx can be:
      constant int, constant list/tuple of integers



Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   df.iloc[0, 0] = df.iloc[2,2]
    ...   df.iloc[1, [1,2]] = df.iloc[0, 1]
    ...   df["D"] = df.iloc[0]
    ...   return df
       A  B  C  D
    0  9  4  7  7
    1  2  4  4  4
    2  3  6  9  9

* :meth:`pandas.DataFrame.insert` ``(loc, column, value, allow_duplicates=False)``
  Supported Arguments:
    * loc
        - constant integer
    * column
        - constant string
    * value
        - scalar
        - list/tuple
        - pandas/numpy array
        - pandas index types
        - series
    * allow_duplicates
        - constant boolean


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   df.insert(3, "D", [-1,-2,-3])
    ...   return df
      A  B  C  D
    0  1  4  7 -1
    1  2  5  8 -2
    2  3  6  9 -3


* :meth:`pandas.DataFrame.isin` ``(values)``

  Supported Arguments:
    * values
       - DataFrame (must have same indicies) + iterable type
       - Numpy array types
       - Pandas array types
       - List/Tuple
       - Pandas Index Types (excluding interval Index and MultiIndex)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   isin_1 = df.isin([1,5,9])
    ...   isin_2 = df.isin(pd.DataFrame({"A": [4,5,6], "C": [7,8,9]}))
    ...   formated_out = "\n".join([isin_1.to_string(), isin_2.to_string()])
    ...   return formated_out
          A      B      C
    0  True   False  False
    1  False  True   False
    2  False  False  True
          A      B     C
    0  False  False  True
    1  False  False  True
    2  False  False  True

.. note::

    DataFrame isin ignores DataFrame indicies. For example. ::

      >>> @bodo.jit
      ... def f():
      ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
      ...   return df.isin(pd.DataFrame({"A": [1,2,3]}, index=["A", "B", "C"]))
            A      B      C
      0  True  False  False
      1  True  False  False
      2  True  False  False

      >>> def f():
      ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
      ...   return df.isin(pd.DataFrame({"A": [1,2,3]}, index=["A", "B", "C"]))
             A      B      C
      0  False  False  False
      1  False  False  False
      2  False  False  False



* :meth:`pandas.DataFrame.itertuples` ``(index=True, name='Pandas')``
    Supported Arguments:
      none

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   for x in pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}).itertuples():
    ...      print(x)
    ...      print(x[0])
    ...      print(x[2:])
    Pandas(Index=0, A=1, B=4, C=7)
    0
    (4, 7)
    Pandas(Index=1, A=2, B=5, C=8)
    1
    (5, 8)
    Pandas(Index=2, A=3, B=6, C=9)
    2
    (6, 9)


* :meth:`pandas.DataFrame.query` ``(expr, inplace=False, **kwargs)``

    Supported Arguments:
      * expr
        - Constant String

Example Usage::

    >>> @bodo.jit
    ... def f(a):
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.query('A > @a')
    >>> f(1)
       A  B  C
    1  2  5  8
    2  3  6  9

.. note::
    * The output of the query must evaluate to a 1d boolean array.
    * Cannot refer to the index by name in the query string.
    * Query must be one line.
    * If using environment variables, they should be passed as arguments to the function.


* :meth:`pandas.DataFrame.tail` ``(n=5)``

  Supported Arguments:
    * n
       - Integer

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   return pd.DataFrame({"A": np.arange(1000)}).tail(3)
          A
    997  997
    998  998
    999  999



Function application, GroupBy & Window:

* :meth:`pandas.DataFrame.apply` ``(func, axis=0, raw=False, result_type=None, args=(), _bodo_inline=False, **kwargs)``

  Supported Arguments:
    * func
       - **Must be constant at Compile Time**
       - function (e.g. lambda) (axis must = 1)
       - jit function (axis must = 1)
       - String which refers to a support DataFrame method
    * axis
       - **Must be constant at Compile Time**
       - Integer (0, 1)
       - String (only if the method takes axis as an argument )
    * _bodo_inline
       - **Must be constant at Compile Time**
       - Boolean

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.apply(lambda x: x["A"] * (x["B"] + x["C"]))
    0    11
    1    26
    2    45
    dtype: int64


.. note ::

    Supports extra `_bodo_inline` boolean argument to manually control bodo's inlining behavior.
    Inlining user-defined functions (UDFs) can potentially improve performance at the expense of
    extra compilation time. Bodo uses heuristics to make a decision automatically if `_bodo_inline` is not provided.

* :meth:`pandas.DataFrame.groupby` ``(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)``

    Supported Arguments:
      * by
         - **Must be constant at Compile Time**
         - String column label
         - List/Tuple of column labels
      * as_index
         - **Must be constant at Compile Time**
         - Boolean
      * dropna
         - **Must be constant at Compile Time**
         - Boolean



.. note ::
  `sort=False` and `observed=True` are set by default. These are the only support values for sort and observed. For more information on using groupby, see :ref:`the groupby Section <pd_groupby_section>`.


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,1,2,2], "B": [-2,-2,2,2]})
    ...   return df.groupby("A").sum()
       B
    A
    1 -4
    2  4


* :meth:`pandas.DataFrame.rolling` ``(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')``

    Supported Arguments:
      * window
         - Integer
         - String (must be parsable as a time offset)
         - datetime.timedelta
         - pd.Timedelta
         - List/Tuple of column labels
      * min_periods
         - Integer
      * center
         - Boolean
      * on
         - **Must be constant at Compile Time**
         - Scalar column label
      * dropna
         - **Must be constant at Compile Time**
         - Boolean


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3,4,5]})
    ...   return df.rolling(3,center=True).mean()
         A
    0  NaN
    1  2.0
    2  3.0
    3  4.0
    4  NaN

For more information, please see :ref:`the Window section <pd_window_section>`.


Computations / Descriptive Stats:

* :meth:`pandas.DataFrame.abs` ``()``

Only supported for dataframes containing numerical data and Timedeltas

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,-2], "B": [3.1,-4.2], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(-10, unit="D")]})
    ...   return df.abs()
       A    B       C
    0  1  3.1 10 days
    1  2  4.2 10 days

* :meth:`pandas.DataFrame.corr` ``(method='pearson', min_periods=1)``
Supported Arguments:
   * min_periods
           - Integer

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [.9, .8, .7, .4], "B": [-.8, -.9, -.8, -.4], "c": [.7, .7, .7, .4]})
    ...   return df.corr()
              A         B        c
    A  1.000000 -0.904656  0.92582
    B -0.904656  1.000000 -0.97714
    c  0.925820 -0.977140  1.00000

* :meth:`pandas.DataFrame.count` ``(axis=0, level=None, numeric_only=False)``

Supported Arguments:
  none

Example Usage::
    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1, None, 3], "B": [None, 2, None]})
    ...   return df.count()
    A    2
    B    1

* :meth:`pandas.DataFrame.cov` ``(min_periods=None, ddof=1)``

Supported Arguments:
  * min_periods
      - Integer


Example Usage::
    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [0.695, 0.478, 0.628], "B": [-0.695, -0.478, -0.628], "C": [0.07, -0.68, 0.193]})
    ...   return df.cov()
              A         B         C
    A  0.012346 -0.012346  0.047577
    B -0.012346  0.012346 -0.047577
    C  0.047577 -0.047577  0.223293



* :meth:`pandas.DataFrame.cumprod` ``(axis=None, skipna=True)``

Supported Arguments:
  None


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1, 2, 3], "B": [.1,np.NaN,12.3],})
    ...   return df.cumprod()
       A    B
    0  1  0.1
    1  2  NaN
    2  6  NaN


.. note::
  Not supported for dataframe with nullable integer.


* :meth:`pandas.DataFrame.cumsum` ``(axis=None, skipna=True)``

Supported Arguments:
  None

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1, 2, 3], "B": [.1,np.NaN,12.3],})
    ...   return df.cumsum()
       A    B
    0  1  0.1
    1  3  NaN
    2  6  NaN

.. note::
  Not supported for dataframe with nullable integer.

* :meth:`pandas.DataFrame.describe` ``(percentiles=None, include=None, exclude=None, datetime_is_numeric=False)``


Supported Arguments:
  None

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [pd.Timestamp(2000, 10, 2), pd.Timestamp(2001, 9, 5), pd.Timestamp(2002, 3, 11)]})
    ...   return df.describe()
            A                    B
    count  3.0                    3
    mean   2.0  2001-07-16 16:00:00
    min    1.0  2000-10-02 00:00:00
    25%    1.5  2001-03-20 00:00:00
    50%    2.0  2001-09-05 00:00:00
    75%    2.5  2001-12-07 12:00:00
    max    3.0  2002-03-11 00:00:00
    std    1.0                  NaN

.. note::
  Only supported for dataframes containing numeric data, and datetime data. Datetime_is_numeric defaults to True in JIT code.

* :meth:`pandas.DataFrame.diff` ``(periods=1, axis=0)``

Supported Arguments:
  * periods
     - Integer

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [pd.Timestamp(2000, 10, 2), pd.Timestamp(2001, 9, 5), pd.Timestamp(2002, 3, 11)]})
    ...   return df.diff(1)
         A        B
    0  NaN      NaT
    1  1.0 338 days
    2  1.0 187 days

.. note::
  Only supported for dataframes containing float, non-null int, and datetime64ns values


* :meth:`pandas.DataFrame.max` ``(axis=None, skipna=None, level=None, numeric_only=None)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.max(axis=1)
    0    7
    1    8
    2    9

.. note::
  Only supported for dataframes containing float, non-null int, and datetime64ns values.

* :meth:`pandas.DataFrame.mean` ``axis=None, skipna=None, level=None, numeric_only=None)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.mean(axis=1)
    0    4.0
    1    5.0
    2    6.0

.. note::
  Only supported for dataframes containing float, non-null int, and datetime64ns values.


* :meth:`pandas.DataFrame.median` ``axis=None, skipna=None, level=None, numeric_only=None)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.median(axis=1)
    0    4.0
    1    5.0
    2    6.0

.. note::
  Only supported for dataframes containing float, non-null int, and datetime64ns values.

* :meth:`pandas.DataFrame.min`

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.min(axis=1)
    0    1
    1    2
    2    3

.. note::
  Only supported for dataframes containing float, non-null int, and datetime64ns values.

* :meth:`pandas.DataFrame.nunique` ``(axis=0, dropna=True)``

Supported Arguments:
  * dropna
     - boolean

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [1,1,1], "C": [4, None, 6]})
    ...   return df.nunique()
    A    3
    B    1
    C    2

* :meth:`pandas.DataFrame.pct_change` ``(periods=1, fill_method='pad', limit=None, freq=None)``

Supported Arguments:
  * periods
     - Integer


Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [10,100,1000,10000]})
    ...   return df.pct_change()
        A
    0  NaN
    1  9.0
    2  9.0
    3  9.0


* :meth:`pandas.DataFrame.pipe` ``(func, *args, **kwargs)``

Supported Arguments:
    * func:

      - JIT function or callable defined within a JIT function.

    * Additional arguments for ``func`` can be passed as additional arguments.

.. note::

    `func` cannot be a tuple

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [10,100,1000,10000]})
    ...   return df.pipe()
    ...
    ... def g(df, axis):
    ...   return df.max(axis)
    ...
    ... f()
    A    3
    dtype: int64


* :meth:`pandas.DataFrame.prod` ``(axis=None, skipna=None, level=None, numeric_only=None)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.prod(axis=1)
    A      6
    B    120
    C    504
    dtype: int64


* :meth:`pandas.DataFrame.product` ``(axis=None, skipna=None, level=None, numeric_only=None)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.product(axis=1)
    A      6
    B    120
    C    504
    dtype: int64

* :meth:`pandas.DataFrame.quantile` ``(q=0.5, axis=0, numeric_only=True, interpolation='linear')``

Supported Arguments:
  * q
     - Float or Int, must be 0<= q <= 1
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.quantile()
    A    2.0
    B    5.0
    C    8.0
    dtype: float64
    dtype: int64

* :meth:`pandas.DataFrame.std` ``(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.std(axis=1)
    0    3.0
    1    3.0
    2    3.0
    dtype: float64

* :meth:`pandas.DataFrame.sum` ``(axis=None, skipna=None, level=None, numeric_only=None, min_count=0)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.sum(axis=1)
    0    12
    1    15
    2    18
    dtype: int64


* :meth:`pandas.DataFrame.var` ``(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)``

Supported Arguments:
  * axis
     - **Must be constant at Compile Time**
     - Integer (0 or 1)

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.var(axis=1)
    0    9.0
    1    9.0
    2    9.0
    dtype: float64


* :meth:`pandas.DataFrame.memory_usage` ``(index=True, deep=False)``

Supported Arguments:
  * index
     - Boolean

Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": np.array([1,2,3], dtype=np.int64), "B": np.array([1,2,3], dtype=np.int32), "C": ["1", "2", "3456689"]})
    ...   return df.memory_usage()
    Index    24
    A        24
    B        12
    C        42
    dtype: int64


Reindexing / Selection / Label manipulation:

* :meth:`pandas.DataFrame.drop` ``(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')``

  *  Only dropping columns supported, either using `columns` argument or setting `axis=1` and using the `labels` argument
  * `labels` and `columns` require constant string, or constant list/tuple of string values
  * `inplace` supported with a constant boolean value
  * All other arguments are unsupported

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   df.drop(columns = ["B", "C"], inplace=True)
    ...   return df
       A
    0  1
    1  2
    2  3


* :meth:`pandas.DataFrame.drop_duplicates` ``(subset=None, keep='first', inplace=False, ignore_index=False)``

Supported Arguments:
  * subset
     - Constant list/tuple of String column names
     - Constant list/tuple of Integer column names
     - Constant String column names
     - Constant Integer column names

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,1,3,4], "B": [1,1,3,3], "C": [7,8,9,10]})
    ...   return df.drop_duplicates(subset = ["A", "B"])
       A  B   C
    0  1  1   7
    2  3  3   9
    3  4  3  10

* :meth:`pandas.DataFrame.duplicated` ``(subset=None, keep='first')``

Supported Arguments: None

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,1,3,4], "B": [1,1,3,3]})
    ...   return df.duplicated()
    0    False
    1     True
    2    False
    3    False
    dtype: bool


* :meth:`pandas.DataFrame.idxmax` ``(axis=0, skipna=True)``

Supported Arguments: None

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.idxmax()
    A    2
    B    2
    C    2
    dtype: int64

* :meth:`pandas.DataFrame.idxmin` ``(axis=0, skipna=True)``

  Supported Arguments: None

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.idxmax()
    A    0
    B    0
    C    20
    dtype: int64

* :meth:`pandas.DataFrame.rename` ``(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore')``

  Supported Arguments:
    * mapper
      - must be constant dictionary. Can only be used alongside axis=1
    * columns
      - must be constant dictionary
    * axis
      - Can only be used alongside mapper argument
    * copy
      - Boolean
    * inplace
      - must be constant boolean

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.rename(columns={"A": "X", "B":"Y", "C":"Z"})
       X  Y  Z
    0  1  4  7
    1  2  5  8
    2  3  6  9

* :meth:`pandas.DataFrame.reset_index` ``(level=None, drop=False, inplace=False, col_level=0, col_fill='')``

  Supported Arguments:
    * level
       - If specified, must drop all levels.
    * drop
       - Constant Boolean
    * inplace
       - Constant Boolean

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}, index = ["X", "Y", "Z"])
    ...   return df.reset_index()
      index  A  B  C
    0     X  1  4  7
    1     Y  2  5  8
    2     Z  3  6  9


* :meth:`pandas.DataFrame.set_index` ``(keys, drop=True, append=False, inplace=False, verify_integrity=False)``

  Supported Arguments:
    * keys
       - must be a constant string

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}, index = ["X", "Y", "Z"])
    ...   return df.set_index("C")
       A  B
    C
    7  1  4
    8  2  5
    9  3  6


* :meth:`pandas.DataFrame.take` ``(indices, axis=0, is_copy=None)``

  Supported Arguments:
    * indicies
       - scalar Integer
       - Pandas Integer Array
       - Numpy Integer Array
       - Integer Series

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.take(pd.Series([-1,-2]))
       A  B  C
    2  3  6  9
    1  2  5  8


Missing data handling:

* :meth:`pandas.DataFrame.dropna` ``(axis=0, how='any', thresh=None, subset=None, inplace=False)``

  Supported Arguments:
    * how
       - Constant String, either "all" or "any"
    * thresh
       - Integer
    * subset
       - Constant list/tuple of String column names
       - Constant list/tuple of Integer column names
       - Constant String column names
       - Constant Integer column names

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3,None], "B": [4, 5,None, None], "C": [6, None, None, None]})
    ...   df_1 = df.dropna(how="all", subset=["B", "C"])
    ...   df_2 = df.dropna(thresh=3)
    ...   formated_out = "\n".join([df_1.to_string(), df_2.to_string()])
    ...   return formated_out
       A  B     C
    0  1  4     6
    1  2  5  <NA>
       A  B  C
    0  1  4  6



* :meth:`pandas.DataFrame.fillna` ``(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)``

  Supported Arguments:

    * value

      - Must be of the same type as the filled column

    * inplace

      - Constant Boolean

    * method

      - **Must be constant at Compile Time**
      - One of "bfill", "backfill", "ffill" , or "pad"

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3,None], "B": [4, 5,None, None], "C": [6, None, None, None]})
    ...   return df.fillna(-1)

* :meth:`pandas.DataFrame.replace` ``(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')``

  Supported Arguments:
    * to_replace
       - Required argumnet
    * value
       - Must be of the same type as to_replace

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.replace(1, -1)
       A  B  C
    0 -1  4  7
    1  2  5  8
    2  3  6  9

Reshaping, sorting, transposing:

* :meth:`pandas.DataFrame.pivot_table` ``(values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False, sort=True)``


  Supported Arguments:
    * values
       - String Constant (required)
    * index
       - String Constant (required)
    * columns
       - String Constant (required)
    * aggfunc
       - String Constant


.. note::
  Annotation of pivot values is required. For example, `@bodo.jit(pivots={'pt': ['small', 'large']})` declares the output pivot table `pt` will have columns called `small` and `large`.

  Example Usage::

    >>> @bodo.jit(pivots={'pivoted_tbl': ['X', 'Y']})
    ... def f():
    ...   df = pd.DataFrame({"A": ["X","X","X","X","Y","Y"], "B": [1,2,3,4,5,6], "C": [10,11,12,20,21,22]})
    ..    pivoted_tbl = df.pivot_table(columns="A", index="B", values="C", aggfunc="mean")
    ...   return pivoted_tbl
          X     Y
    B
    1  10.0   NaN
    2  11.0   NaN
    3  12.0   NaN
    4  20.0   NaN
    5   NaN  21.0
    6   NaN  22.0


* :meth:`pandas.DataFrame.sample` ``(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None, ignore_index=False)``

    Supported Arguments:
      * n
         - Integer
      * frac
         - Float
      * replace
         - boolean


  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
    ...   return df.sample(1)
       A  B  C
    2  3  6  9


* :meth:`pandas.DataFrame.sort_index` ``(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)``

    Supported Arguments:
      * ascending
         - boolean
      * na_position
         - constant String ("first" or "last")



  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3]}, index=[1,None,3])
    ...   return df.sort_index(ascending=False, na_position="last")
         A
    3    3
    1    1
    NaN  2


* :meth:`pandas.DataFrame.sort_values` ``(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)``

    Supported Arguments:
      * by
         - constant String or constant list of strings
      * ascending
         - boolean
         - list/tuple of boolean, with length equal to the number of key columns
      * inplace
         - Constant Boolean
      * na_position
         - constant String ("first" or "last")
         - constant list/tuple of String, with length equal to the number of key columns


  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,2,None], "B": [4, 5, 6, None]})
    ...   df.sort_values(by=["A", "B"], ascending=[True, False], na_position=["first", "last"], inplace=True)
    ...   return df
          A     B
    3  <NA>  <NA>
    0     1     4
    2     2     6
    1     2     5


* :meth:`pandas.DataFrame.to_string` ``(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, min_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None, max_colwidth=None, encoding=None)``

(not optimized)

    Supported Arguments:
      * buf
      * columns
      * col_space
      * header
      * index*
      * na_rep
      * formatters
      * float_format
      * sparsify
      * index_names
      * justify
      * max_rows
      * min_rows
      * max_cols
      * how_dimensions
      * decimal
      * line_width
      * max_colwidth
      * encoding


  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3]})
    ...   return df.to_string()
       A
    0  1
    1  2
    2  3

.. note::
   When called on a dsitributed dataframe, the string returned for each rank will be reflective of the dataframe for that rank.

Combining / joining / merging:

* :meth:`pandas.DataFrame.append` ``(other, ignore_index=False, verify_integrity=False, sort=False)``

  Supported Arguments:
    * other
     - Dataframe
     - list/tuple of Dataframe
    * ignore_index
     - constant Boolean

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
    ...   return df.append(pd.DataFrame({"A": [-1,-2,-3], "C": [4,5,6]}))
       A    B    C
    0  1  4.0  NaN
    1  2  5.0  NaN
    2  3  6.0  NaN
    0 -1  NaN  4.0
    1 -2  NaN  5.0
    2 -3  NaN  6.0


* :meth:`pandas.DataFrame.assign` ``(**kwargs)``

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
    ...   df2 = df.assign(C = 2 * df["B"], D = lambda x: x.C * -1)
    ...   return df2
       A  B   C   D
    0  1  4   8  -8
    1  2  5  10 -10
    2  3  6  12 -12


.. note::
    arguments can be JIT functions, lambda functions, or values that can be used to initialize a Pandas Series.

* :meth:`pandas.DataFrame.join` ``(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)``

  Supported Arguments:
    * other
       - Dataframe
    * on
       - constant string column name
       - constant list/tuple of column names

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
    ...   return df.join(on = "A", other=pd.DataFrame({"C": [-1,-2,-3], "D": [4,5,6]}))
       A  B     C     D
    0  1  4    -2     5
    1  1  5    -2     5
    2  3  6  <NA>  <NA>


.. note::
   Joined dataframes cannot have common columns. The output dataframe is not sorted by default for better parallel performance


* :meth:`pandas.DataFrame.merge` ``(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)``

See :ref:`pd.merge <pd_merge_fn>` for full list of support arguments, and more examples.

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
    ...   return df.merge(pd.DataFrame({"C": [-1,-2,-3], "D": [4,4,6]}), left_on = "B", right_on = "D")
       A  B  C  D
    0  1  4 -1  4
    1  1  4 -2  4
    2  3  6 -3  6


Time series-related:

* :meth:`pandas.DataFrame.shift` ``(periods=1, freq=None, axis=0, fill_value=NoDefault.no_default)``

  Supported Arguments:
    * periods
      - Integer

  Example Usage::

    >>> @bodo.jit
    ... def f():
    ...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
    ...   return df.shift(1)
         A    B
    0  NaN  NaN
    1  1.0  4.0
    2  1.0  5.0

.. note::
  Only soupported for dataframes containing numeric, boolean, datetime.date and string types.



.. _pandas-f-out:

Serialization / IO / conversion:

Also see :ref:`S3` and :ref:`HDFS` configuration requirements and more on :ref:`file_io`.

* :meth:`pandas.DataFrame.to_csv`
  * ``compression`` argument defaults to ``None`` in JIT code. This is the only supported value of this argument.
  * ``mode`` argument supports only the default value ``"w"``.
  * ``errors`` argument supports only the default value ``strict``.
  * ``storage_options`` argument supports only the default value ``None``.
* :meth:`pandas.DataFrame.to_json`
* :meth:`pandas.DataFrame.to_parquet`
* :meth:`pandas.DataFrame.to_sql`
  * :ref:`example usage and more system specific instructions <sql-section>`
  * Argument ``con`` is supported but only as a string form. SQLalchemy `connectable` is not supported.
  * Argument ``name``, ``schema``, ``if_exists``, ``index``, ``index_label``, ``dtype``, ``method`` are supported.
  * Argument ``chunksize`` is not supported.

Plotting

* :meth:`pandas.DataFrame.plot` ``(x=None, y=None, kind="line", figsize=None, xlabel=None, ylabel=None, title=None, legend=True, fontsize=None, xticks=None, yticks=None, ax=None)``


  Supported Arguments:
    * x
      - Constant String column name
      - Constant Integer
    * y
      - Constant String column name
      - Constant Integer
    * kind
      - constant String ("line" or "scatter")
    * figsize
      - constant numeric tuple (width, height)
    * xlabel
      - constant String
    * ylabel
      - constant String
    * title
      - constant String
    * legend
      - boolean
    * fontsize
    * xticks
      - Constant Tuple
    * yticks
      - Constant Tuple
    * ax




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


Timestamp
~~~~~~~~~

Timestamp functionality is documented in `pandas.Timestamp <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html>`_.

* :class:`pandas.Timestamp` ``(ts_input=<object object>, freq=None, tz=None, unit=None, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, nanosecond=None, tzinfo=None, *, fold=None)``

Supported arguments:
    * ts_input
     - string
     - integer
     - timestamp
     - datetimedate
    * unit
     - constant string
    * year
     - integer
    * month
     - integer
    * day
     - integer
    * hour
     - integer
    * minute
     - integer
    * second
     - integer
    * microsecond
     - integer
    * nanosecond
     - integer

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   return I.copy(name="new_name")
    ...   ts1 = pd.Timestamp('2021-12-09 09:57:44.114123')
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts3 = pd.Timestamp(100, unit="days")
    ...   ts4 = pd.Timestamp(datetime.date(2021, 12, 9), hour = 9, minute=57, second=44, microsecond=114123)
    ...   return (ts1, ts2, ts3, ts4)
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 09:57:44.114123'), Timestamp('1970-04-11 00:00:00'), Timestamp('2021-12-09 09:57:44.114123'))


* :attr:`pandas.Timestamp.day`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day
    9

* :attr:`pandas.Timestamp.hour`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.hour
    9

* :attr:`pandas.Timestamp.microsecond`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.microsecond
    114123

* :attr:`pandas.Timestamp.month`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.month
    month

* :attr:`pandas.Timestamp.nanosecond`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(12, unit="ns")
    ...   return ts2.nanosecond
    12

* :attr:`pandas.Timestamp.second`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.second
    44

* :attr:`pandas.Timestamp.year`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.year
    2021

* :attr:`pandas.Timestamp.dayofyear`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.dayofyear
    343
* :attr:`pandas.Timestamp.day_of_year`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_year
    343
* :attr:`pandas.Timestamp.dayofweek`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_year
    343
* :attr:`pandas.Timestamp.day_of_week`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.day_of_week
    3
* :attr:`pandas.Timestamp.days_in_month`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.days_in_month
    31
* :attr:`pandas.Timestamp.daysinmonth`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return ts2.daysinmonth
    31

* :attr:`pandas.Timestamp.is_leap_year`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2020, month=2,day=2)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   return (ts1.is_leap_year, ts2.is_leap_year)
    (True, False)
* :attr:`pandas.Timestamp.is_month_start`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=2)
    ...   return (ts1.is_moth_start, ts2.is_moth_start)
    (True, False)

* :attr:`pandas.Timestamp.is_month_end`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=30)
    ...   return (ts1.is_moth_end, ts2.is_moth_end)
    (True, False)

* :attr:`pandas.Timestamp.is_quarter_start`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
    ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
    ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
    (False, True)

* :attr:`pandas.Timestamp.is_quarter_end`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=30)
    ...   ts2 = pd.Timestamp(year=2021, month=10, day=1)
    ...   return (ts1.is_quarter_start, ts2.is_quarter_start)
    (True, False)

* :attr:`pandas.Timestamp.is_year_start`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
    ...   return (ts1.is_year_start, ts2.is_year_start)
    (False, True)

* :attr:`pandas.Timestamp.is_year_end`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=31)
    ...   ts2 = pd.Timestamp(year=2021, month=1, day=1)
    ...   return (ts1.is_year_end, ts2.is_year_end)
    (True, False)

* :attr:`pandas.Timestamp.quarter`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=1)
    ...   return (ts1.quarter, ts2.quarter)
    (4, 3)

* :attr:`pandas.Timestamp.week`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
    ...   return (ts1.week, ts2.week)
    (35, 38)

* :attr:`pandas.Timestamp.weekofyear`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=9, day=1)
    ...   ts2 = pd.Timestamp(year=2021, month=9, day=20)
    ...   return (ts1.weekofyear, ts2.weekofyear)
    (35, 38)

* :attr:`pandas.Timestamp.value`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(12345, unit="ns").value
    12345
* :meth:`pandas.Timestamp.ceil` ``(freq, ambiguous='raise', nonexistent='raise')``
Supported arguments:
  * freq
   - string

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-10 00:00:00'))

* :meth:`pandas.Timestamp.date` ``()``

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).date()
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 09:57:44.114123'), datetime.date(2021, 12, 9))


* :meth:`pandas.Timestamp.day_name` ``(*args, **kwargs)``
  Supported arguments:
    None

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   day_1 = pd.Timestamp(year=2021, month=12, day=9).day_name()
    ...   day_2 = pd.Timestamp(year=2021, month=12, day=10).day_name()
    ...   day_3 = pd.Timestamp(year=2021, month=12, day=11).day_name()
    ...   return (day_1, day_2, day_3)
    ('Thursday', 'Friday', 'Saturday')

* :meth:`pandas.Timestamp.floor`
Supported arguments:
  * freq
   - string

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).ceil("D")
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 09:57:44.114123'), Timestamp('2021-12-09 00:00:00'))

* :meth:`pandas.Timestamp.isocalendar`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
    ...   return (ts1, ts2)
    (2021, 49, 4)

* :meth:`pandas.Timestamp.isoformat`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).isocalendar()
    ...   return (ts1, ts2)
    '2021-12-09T09:57:44'

* :meth:`pandas.Timestamp.month_name` ``(*args, **kwargs)``
  Supported arguments:
    None

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9).month_name()
    'December'

* :meth:`pandas.Timestamp.normalize`

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 9, minute=57, second=44, microsecond=114123).normalize()
    ...   return (ts1, ts2)
    Timestamp('2021-12-09 00:00:00')

* :meth:`pandas.Timestamp.round` ``(freq, ambiguous='raise', nonexistent='raise')``
Supported arguments:
  * freq
   - string

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9, hour = 12).round()
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=9, hour = 13).round()
    ...   return (ts1, ts2)
    (Timestamp('2021-12-09 00:00:00'),Timestamp('2021-12-10 00:00:00'))

* :meth:`pandas.Timestamp.strftime` ``(format)``
Supported arguments:
  * format
   - string

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9, hour = 12).strftime('%Y-%m-%d %X')
    '2021-12-09 12:00:00'

* :meth:`pandas.Timestamp.toordinal` ``()``
  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp(year=2021, month=12, day=9).toordinal()
    738133

* :meth:`pandas.Timestamp.weekday` ``()``

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   ts1 = pd.Timestamp(year=2021, month=12, day=9)
    ...   ts2 = pd.Timestamp(year=2021, month=12, day=10)
    ...   return (ts1.weekday(), ts2.weekday())
    (3, 4)

* :classmeth:`pandas.Timedelta.now` ``(tz=None)``
Supported arguments:
  None

  Example Usage ::

    >>> @bodo.jit
    ... def f():
    ...   return pd.Timestamp.now()
    Timestamp('2021-12-10 10:54:06.457168')



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


.. _pd_window_section:

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

.. _pd_groupby_section:

GroupBy
~~~~~~~

* :meth:`pandas.DataFrame.groupby` ``(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)``

Supported Arguments:

  * by (default=None):

    - **Must be constant at Compile Time**
    - **This argument is required**
    - Column label or list of column labels


  * as_index (default=True):

    - **Must be constant at Compile Time**
    - Boolean

  * dropna (default=True):

    - **Must be constant at Compile Time**
    - Boolean


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B", dropna=True, as_index=False).count()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

         B   A   C
    0  421  10  10
    1  f31   5  10



* :meth:`pandas.Series.groupby` ``(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)``

Supported Arguments:

  * by (default=None):

    - **Must be constant at Compile Time**
    - Array-like or Series data. This is not supported with Decimal or Categorical data.


  * level (default=None):

    - **Must be constant at Compile Time**
    - Only ``level=0`` is supported and not with MultiIndex.

  .. important:

    You must provide exactly one of ``by`` and ``level``

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(S, by_series):
    ...     return S.groupby(by_series).count()
    >>> S = pd.Series([1, 2, 24, None] * 5)
    >>> by_series = pd.Series(["421", "f31"] * 10)
    >>> f(S, by_series)

    421    10
    f31     5
    Name: , dtype: int64

.. note::

  Series.groupby doesn't currently keep the name of the original Series.

* :meth:`pandas.core.groupby.GroupBy.apply` ``(func, *args, **kwargs)``

Supported Arguments:

  * func:

    - JIT function or callable defined within a JIT function that returns a DataFrame or Series

  * Additional arguments for ``func`` can be passed as additional arguments.

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df, y):
    ...     return df.groupby("B", dropna=True).apply(lambda group, y: group.sum(axis=1) + y, y=y)
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> y = 4
    >>> f(df, y)

    B
    421  0          6.510
         2          8.421
         4     233260.000
         6         16.210
         8          6.510
         10         8.421
         12    233260.000
         14        16.210
         16         6.510
         18         8.421
    f31  1     233260.000
         3         16.210
         5          6.510
         7          8.421
         9     233260.000
         11        16.210
         13         6.510
         15         8.421
         17    233260.000
         19        16.210
    dtype: float64



* :meth:`pandas.core.groupby.GroupBy.agg` ``(func, *args, **kwargs)``

Supported Arguments:

  * func:

    - JIT function or callable defined within a JIT function or a constant dictionary
      mapping column name to a function

    .. note:

      Passing a list of functions is also supported if only one output column is selected.

  * Additional arguments for ``func`` can be passed as additional arguments.

  .. note:

      Output column names can be specified using keyword arguments and `pd.NamedAgg()`.

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B", dropna=True).agg({"A": lambda x: max(x)})
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

            A
    B
    421  24.0
    f31   2.0


* :meth:`pandas.core.groupby.DataFrameGroupBy.aggregate` ``(func, *args, **kwargs)``

Supported Arguments:

  * func:

    - JIT function or callable defined within a JIT function or a constant dictionary
      mapping column name to a function

    .. note:

      Passing a list of functions is also supported if only one output column is selected.

  * Additional arguments for ``func`` can be passed as additional arguments.

  .. note:

      Output column names can be specified using keyword arguments and `pd.NamedAgg()`.

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B", dropna=True).agg({"A": lambda x: max(x)})
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

            A
    B
    421  24.0
    f31   2.0


* :meth:`pandas.core.groupby.DataFrameGroupBy.transform` ``(func, *args, engine=None, engine_kwargs=None, **kwargs)``

  Supported Arguments:

    * func:

      - Either a constant string or a Python function from the builtins
        module that matches a supported operation. Numpy functions
        cannot be provided.

      .. note:

        The supported builtin functions are `'count'`, `'first'`, `'last'`,
        `'min'`, `'max'`, `'mean'`, `'median'`, `'nunique'`, `'prod'`,
        `'std'`, `'sum'`, and `'var'`

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B", dropna=True).transform(max)
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

           A          C
    0   24.0  233232.00
    1    2.0      12.21
    2   24.0  233232.00
    3    2.0      12.21
    4   24.0  233232.00
    5    2.0      12.21
    6   24.0  233232.00
    7    2.0      12.21
    8   24.0  233232.00
    9    2.0      12.21
    10  24.0  233232.00
    11   2.0      12.21
    12  24.0  233232.00
    13   2.0      12.21
    14  24.0  233232.00
    15   2.0      12.21
    16  24.0  233232.00
    17   2.0      12.21
    18  24.0  233232.00
    19   2.0      12.21


* :meth:`pandas.core.groupby.GroupBy.pipe` ``(func, *args, **kwargs)``

  Supported Arguments:

    * func:

      - JIT function or callable defined within a JIT function.

    * Additional arguments for ``func`` can be passed as additional arguments.


  .. note::

    `func` cannot be a tuple

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df, y):
    ...     return df.groupby("B").pipe(lambda grp, y: grp.sum() - y, y=y)
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> y = 5
    >>> f(df, y)

             A            C
    B
    421  120.0  1166162.550
    f31    5.0       68.155



* :meth:`pandas.core.groupby.GroupBy.count` ``()``

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").count()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

          A   C
    B
    421  10  10
    f31   5  10

* :meth:`pandas.core.groupby.GroupBy.cumsum` ``(axis=0)``

  .. note::

    cumsum is only supported on numeric columns and is not supported on boolean columns

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").cumsum()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

            A            C
    0     1.0        1.510
    1     2.0        2.421
    2    25.0   233233.510
    3     NaN       14.631
    4    26.0   233235.020
    5     4.0       17.052
    6    50.0   466467.020
    7     NaN       29.262
    8    51.0   466468.530
    9     6.0       31.683
    10   75.0   699700.530
    11    NaN       43.893
    12   76.0   699702.040
    13    8.0       46.314
    14  100.0   932934.040
    15    NaN       58.524
    16  101.0   932935.550
    17   10.0       60.945
    18  125.0  1166167.550
    19    NaN       73.155



* :meth:`pandas.core.groupby.GroupBy.first` ``(numeric_only=False, min_count=-1)``

.. note::

    first is not supported on columns with nested array types


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").first()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

           A      C
    B
    421  1.0  1.510
    f31  2.0  2.421


* :meth:`pandas.core.groupby.GroupBy.head` ``(n=5)``


Supported Arguments:

  * n (default=5)

    - **Must be constant at Compile Time**
    - Non-negative integer


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").head()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

          A    B           C
    0   1.0  421       1.510
    1   2.0  f31       2.421
    2  24.0  421  233232.000
    3   NaN  f31      12.210
    4   1.0  421       1.510
    5   2.0  f31       2.421
    6  24.0  421  233232.000
    7   NaN  f31      12.210
    8   1.0  421       1.510
    9   2.0  f31       2.421

* :meth:`pandas.core.groupby.GroupBy.last` ``(numeric_only=False, min_count=-1)``

  .. note::

    last is not supported on columns with nested array types


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").last()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

            A          C
    B
    421  24.0  233232.00
    f31   2.0      12.21


* :meth:`pandas.core.groupby.GroupBy.max` ``(numeric_only=False, min_count=-1)``

  .. note::

    * max is not supported on columns with nested array types.
    * Categorical columns must be ordered.


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").max()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

            A          C
    B
    421  24.0  233232.00
    f31   2.0      12.21


* :meth:`pandas.core.groupby.GroupBy.mean` ``(numeric_only=NoDefault.no_default)``

  .. note::

    mean is only supported on numeric columns and is not supported on boolean column


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").mean()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

            A            C
    B
    421  12.5  116616.7550
    f31   2.0       7.3155


* :meth:`pandas.core.groupby.GroupBy.median` ``(numeric_only=NoDefault.no_default)``

  .. note::

    median is only supported on numeric columns and is not supported on boolean column


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").median()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

            A            C
    B
    421  12.5  116616.7550
    f31   2.0       7.3155


* :meth:`pandas.core.groupby.GroupBy.min` ``(numeric_only=False, min_count=-1)``

  .. note::

    * min is not supported on columns with nested array types
    * Categorical columns must be ordered.


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").min()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

           A      C
    B
    421  1.0  1.510
    f31  2.0  2.421


* :meth:`pandas.core.groupby.GroupBy.prod` ``(numeric_only=NoDefault.no_default, min_count=0)``

  .. note::

    prod is not supported on columns with nested array types


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").prod()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

                 A             C
    B
    421  7962624.0  5.417831e+27
    f31       32.0  2.257108e+07


* :meth:`pandas.core.groupby.GroupBy.rolling` ``(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')``


Supported Arguments:

  * window:

    - Integer, String, Datetime, or Timedelta value

  * min_periods (default=None):

    - Integer

  * center (default=False):

    - Boolean

  * on (default=None):

    - **Must be constant at Compile Time**
    - Column label


  .. note::

    This is equivalent to performing the DataFrame API
    on each groupby. All operations of the rolling API
    can be used with groupby.

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").rolling(2).mean
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

               A            C
    B
    421 0    NaN          NaN
        2    NaN          NaN
        4   12.5  116616.7550
        6    NaN       7.3155
        8   12.5  116616.7550
        10   NaN       7.3155
        12  12.5  116616.7550
        14   NaN       7.3155
        16  12.5  116616.7550
        18   NaN       7.3155
    f31 1   12.5  116616.7550
        3    NaN       7.3155
        5   12.5  116616.7550
        7    NaN       7.3155
        9   12.5  116616.7550
        11   NaN       7.3155
        13  12.5  116616.7550
        15   NaN       7.3155
        17  12.5  116616.7550
        19   NaN       7.3155


* :meth:`pandas.core.groupby.GroupBy.size` ``()``


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").size()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

    B
    421    10
    f31    10
    dtype: int64


* :meth:`pandas.core.groupby.GroupBy.std` ``(ddof=1)``

  .. note::

    std is only supported on numeric columns and is not supported on boolean column


Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").std()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

                 A              C
    B
    421  12.122064  122923.261366
    f31   0.000000       5.159256



* :meth:`pandas.core.groupby.GroupBy.sum` ``(numeric_only=NoDefault.no_default, min_count=0)``

  .. note::

    sum is not supported on columns with nested array types

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").sum()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

             A            C
    B
    421  125.0  1166167.550
    f31   10.0       73.155

* :meth:`pandas.core.groupby.GroupBy.var` ``(ddof=1)``

  .. note::

    var is only supported on numeric columns and is not supported on boolean column

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").var()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

                  A             C
    B
    421  146.944444  1.511013e+10
    f31    0.000000  2.661792e+01


* :meth:`pandas.core.groupby.DataFrameGroupBy.idxmax` ``(axis=0, skipna=True)``

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").idxmax()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

         A  C
    B
    421  2  2
    f31  1  3


* :meth:`pandas.core.groupby.DataFrameGroupBy.idxmin` ``(axis=0, skipna=True)``

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").idxmin()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

         A  C
    B
    421  0  0
    f31  1  1

* :meth:`pandas.core.groupby.DataFrameGroupBy.nunique` ``(dropna=True)``

Supported Arguments:

  * dropna (default=True):

    - Boolean

  .. note::

    nunique is not supported on columns with nested array types

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").nunique()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

         A  C
    B
    421  2  2
    f31  1  2

* :meth:`pandas.core.groupby.DataFrameGroupBy.shift` ``(periods=1, freq=None, axis=0, fill_value=None)``

  .. note::

    shift is not supported on columns with nested array types

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(df):
    ...     return df.groupby("B").shift()
    >>> df = pd.DataFrame(
    ...      {
    ...          "A": [1, 2, 24, None] * 5,
    ...          "B": ["421", "f31"] * 10,
    ...          "C": [1.51, 2.421, 233232, 12.21] * 5
    ...      }
    ... )
    >>> f(df)

           A           C
    0    NaN         NaN
    1    NaN         NaN
    2    1.0       1.510
    3    2.0       2.421
    4   24.0  233232.000
    5    NaN      12.210
    6    1.0       1.510
    7    2.0       2.421
    8   24.0  233232.000
    9    NaN      12.210
    10   1.0       1.510
    11   2.0       2.421
    12  24.0  233232.000
    13   NaN      12.210
    14   1.0       1.510
    15   2.0       2.421
    16  24.0  233232.000
    17   NaN      12.210
    18   1.0       1.510
    19   2.0       2.421

* :meth:`pandas.core.groupby.SeriesGroupBy.value_counts` ``(normalize=False, sort=True, ascending=False, bins=None, dropna=True)``

Supported Arguments:

  * ascending (default=False):

    - **Must be constant at Compile Time**
    - Boolean

Example Usage:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(S):
    ...     return S.groupby(level=0).value_counts()
    >>> S = pd.Series([1, 2, 24, None] * 5, index = ["421", "f31"] * 10)
    >>> f(S)

    421  1.0     5
         24.0    5
    f31  2.0     5
    Name: , dtype: int64


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
