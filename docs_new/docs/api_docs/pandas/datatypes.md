# Data Types {#pandas-dtype}

Bodo supports the following data types as values in Pandas Dataframe and
Series data structures. This represents all [Pandas data
types](https://pandas.pydata.org/pandas-docs/stable/reference/arrays.html){target="blank"}
except `TZ-aware datetime`, `Period`,
`Interval`, and `Sparse` (which will be
supported in the future). Comparing to Spark, equivalents of all [Spark
data types](http://spark.apache.org/docs/latest/sql-ref-datatypes.html){target="blank"}
are supported.
  
 - Numpy booleans: `np.bool_`.
 - Numpy integer data types: `np.int8`, `np.int16`, `np.int32`, `np.int64`, `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`.
 - Numpy floating point data types: `np.float32`, `np.float64`.
 - Numpy datetime data types: `np.dtype("datetime64[ns]")` and `np.dtype("timedelta[ns]")`. The resolution has to be `ns` currently, which covers most practical use cases.
 - Numpy complex data types: `np.complex64` and `np.complex128`.
 - Strings (including nulls).
 - `datetime.date` values (including nulls).
 - `datetime.timedelta` values (including nulls).
 - Pandas [nullable integers](https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html){target=blank"}.
 - Pandas [nullable booleans](https://pandas.pydata.org/pandas-docs/stable/user_guide/boolean.html){target=blank"}.
 - Pandas [Categoricals](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html){target=blank"}.
 - Lists of other data types.
 - Tuples of other data types.
 - Structs of other data types.
 - Maps of other data types (each map is a set of key-value pairs). All keys should have the same type to ensure type stability. All values should have the same type as well.
 - `decimal.Decimal` values (including nulls). The decimal values are stored as fixed-precision [Apache Arrow Decimal128](https://arrow.apache.org/docs/cpp/api/utilities.html#classarrow_1_1_decimal128) format, which is also similar to [PySpark decimals](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html). The decimal type has a `precision` (the maximum total number of digits) and a `scale` (the number of digits on the right of dot) attribute, specifying how the stored data is interpreted. For example, the (4, 2) case can store from -999.99 to 999.99. The precision can be up to 38, and the scale must be less or equal to precision. Arbitrary-precision Python `decimal.Decimal` values are converted with precision of 38 and scale of 18.

In addition, it may be desirable to specify type annotations in some
cases (see [Specifying I/O Data Types Manually][non-constant-filepaths] for example).
Typically these types are array types and they all can be
accessed directly from the `bodo` module. The following
table can be used to select the necessary Bodo Type based upon the
desired Python, Numpy, or Pandas type.

| Bodo Type Name| Equivalent Python, Numpy, or Pandas type|
|---------------|-----------------------------------------|
| `bodo.bool_[:]`, `bodo.int8[:]`, ..., `bodo.int64[:]`, `bodo.uint8[:]`, ..., `bodo.uint64[:]`, `bodo.float32[:]`, `bodo.float64[:]` | One-dimensional Numpy array of the given type. A full list of supported Numpy types can be found [here](https://numba.readthedocs.io/en/stable/reference/types.html#numbers){target="blank"}. A multidimensional can be specified by adding additional colons (*e.g.*, `bodo.int32[:, :, :]` for a three-dimensional array).|
| `bodo.string_array_type`| Array of nullable strings|
| `bodo.IntegerArrayType(integer_type)`|  Array of Pandas nullable integers of the given integer type. <br> *e.g.*, `bodo.IntegerArrayType(bodo.int64)`|
| `bodo.boolean_array`| Array of Pandas nullable booleans|
| `bodo.datetime64ns[:]`| Array of Numpy datetime64 values|
| `bodo.timedelta64ns[:]`|Array of Numpy timedelta64 values|
| `bodo.datetime_date_array_type`|Array of datetime.date types|
| `bodo.datetime_timedelta_array_type`|Array of datetime.timedelta types|
| `bodo.DecimalArrayType(precision, scale)`| Array of Apache Arrow Decimal128 values with the given precision and scale. <br> *e.g.*, `bodo.DecimalArrayType(38, 18)`|
| `bodo.binary_array_type`|Array of nullable bytes values|
| `bodo.StructArrayType(data_types, field_names)`| Array of a user defined struct with the given tuple of data types and field names. <br> *e.g.*, `bodo.StructArrayType((bodo.int32[:], bodo.datetime64ns[:]), ("a", "b"))`|
| `bodo.TupleArrayType(data_types)`| Array of a user defined tuple with the given tuple of data types. <br> *e.g.*, `bodo.TupleArrayType((bodo.int32[:], bodo.datetime64ns[:]))`|
| `bodo.MapArrayType(key_arr_type, value_arr_type)`| Array of Python dictionaries with the given key and value array types. <br> *e.g.*, `bodo.MapArrayType(bodo.uint16[:], bodo.string_array_type)`|
| `bodo.PDCategoricalDtype(cat_tuple, cat_elem_type, is_ordered_cat)`| Pandas categorical type with the possible categories, each category's type, and if the categories are ordered. <br> *e.g.*, `bodo.PDCategoricalDtype(("A", "B", "AA"), bodo.string_type, True)`|
| `bodo.CategoricalArrayType(categorical_type)`| Array of Pandas categorical values. <br> *e.g.*, `bodo.CategoricalArrayType(bodo.PDCategoricalDtype(("A", "B", "AA"), bodo.string_type, True))`|
| `bodo.DatetimeIndexType(name_type)`|Index of datetime64 values with a given name type. <br> *e.g.*, `bodo.DatetimeIndexType(bodo.string_type)`|
| `bodo.NumericIndexType(data_type, name_type)`| Index of `pd.Int64`, `pd.Uint64`, or `Float64` objects, based upon the given data_type and name type. <br> *e.g.*, `bodo.NumericIndexType(bodo.float64, bodo.string_type)`|
| `bodo.PeriodIndexType(freq, name_type)`| pd.PeriodIndex with a given frequency and name type. <br> *e.g.*, `bodo.PeriodIndexType('A', bodo.string_type)`|
| `bodo.RangeIndexType(name_type)`| RangeIndex with a given name type. <br> *e.g.*, `bodo.RangeIndexType(bodo.string_type)`|
| `bodo.StringIndexType(name_type)`| Index of strings with a given name type. <br> *e.g.*, `bodo.StringIndexType(bodo.string_type)`|
| `bodo.BinaryIndexType(name_type)`| Index of binary values with a given name type. <br> *e.g.*, `bodo.BinaryIndexType(bodo.string_type)`|
| `bodo.TimedeltaIndexType(name_type)`| Index of timedelta64 values with a given name type.<br> *e.g.*, `bodo.TimedeltaIndexType(bodo.string_type)`|
| `bodo.SeriesType(dtype=data_type, index=index_type, name_typ=name_type)`| Series with a given data type, index type, and name type. <br> *e.g.*, `bodo.SeriesType(bodo.float32, bodo.DatetimeIndexType(bodo.string_type), bodo.string_type)`|
| `bodo.DataFrameType(data_types_tuple, index_type, column_names)`| DataFrame with a tuple of data types, an index type, and the names of the columns. <br> *e.g.*, `bodo.DataFrameType((bodo.int64[::1], bodo.float64[::1]), bodo.RangeIndexType(bodo.none), ("A", "B"))`|

