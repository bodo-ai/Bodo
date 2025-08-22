# Typing Considerations

This section discusses some supported Pandas datatypes, potential typing related issues, and ways to resolve them.

##  Supported Pandas Data Types {#pandas-dtype}

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
cases (*e.g.*, [file I/O array input types][non-constant-filepaths]).
Typically these types are array types and they all can be
accessed directly from the `bodo` module. The following
table can be used to select the necessary Bodo Type based upon the
desired Python, Numpy, or Pandas type.

| Bodo Type Name| Equivalent Python, Numpy, or Pandas type|
|---------------|-----------------------------------------|
| `bodo.types.bool_[:]`, `bodo.types.int8[:]`, ..., `bodo.types.int64[:]`, `bodo.types.uint8[:]`, ..., `bodo.types.uint64[:]`, `bodo.types.float32[:]`, `bodo.types.float64[:]` | One-dimensional Numpy array of the given type. A full list of supported Numpy types can be found [here](https://numba.readthedocs.io/en/stable/reference/types.html#numbers){target="blank"}. A multidimensional can be specified by adding additional colons (*e.g.*, `bodo.types.int32[:, :, :]` for a three-dimensional array).|
| `bodo.types.string_array_type`| Array of nullable strings|
| `bodo.types.IntegerArrayType(integer_type)`|  Array of Pandas nullable integers of the given integer type. <br> *e.g.*, `bodo.types.IntegerArrayType(bodo.types.int64)`|
| `bodo.types.boolean_array_type`| Array of Pandas nullable booleans|
| `bodo.types.datetime64ns[:]`| Array of Numpy datetime64 values|
| `bodo.types.timedelta64ns[:]`|Array of Numpy timedelta64 values|
| `bodo.types.datetime_date_array_type`|Array of datetime.date types|
| `bodo.types.timedelta_array_type`|Array of datetime.timedelta types|
| `bodo.types.DecimalArrayType(precision, scale)`| Array of Apache Arrow Decimal128 values with the given precision and scale. <br> *e.g.*, `bodo.types.DecimalArrayType(38, 18)`|
| `bodo.types.binary_array_type`|Array of nullable bytes values|
| `bodo.types.StructArrayType(data_types, field_names)`| Array of a user defined struct with the given tuple of data types and field names. <br> *e.g.*, `bodo.types.StructArrayType((bodo.types.int32[:], bodo.types.datetime64ns[:]), ("a", "b"))`|
| `bodo.types.TupleArrayType(data_types)`| Array of a user defined tuple with the given tuple of data types. <br> *e.g.*, `bodo.types.TupleArrayType((bodo.types.int32[:], bodo.types.datetime64ns[:]))`|
| `bodo.types.MapArrayType(key_arr_type, value_arr_type)`| Array of Python dictionaries with the given key and value array types. <br> *e.g.*, `bodo.types.MapArrayType(bodo.types.uint16[:], bodo.types.string_array_type)`|
| `bodo.PDCategoricalDtype(cat_tuple, cat_elem_type, is_ordered_cat)`| Pandas categorical type with the possible categories, each category's type, and if the categories are ordered. <br> *e.g.*, `bodo.PDCategoricalDtype(("A", "B", "AA"), bodo.types.string_type, True)`|
| `bodo.CategoricalArrayType(categorical_type)`| Array of Pandas categorical values. <br> *e.g.*, `bodo.CategoricalArrayType(bodo.PDCategoricalDtype(("A", "B", "AA"), bodo.types.string_type, True))`|
| `bodo.DatetimeIndexType(name_type)`|Index of datetime64 values with a given name type. <br> *e.g.*, `bodo.DatetimeIndexType(bodo.types.string_type)`|
| `bodo.NumericIndexType(data_type, name_type)`| Index of `pd.Int64`, `pd.Uint64`, or `Float64` objects, based upon the given data_type and name type. <br> *e.g.*, `bodo.NumericIndexType(bodo.types.float64, bodo.types.string_type)`|
| `bodo.PeriodIndexType(freq, name_type)`| pd.PeriodIndex with a given frequency and name type. <br> *e.g.*, `bodo.PeriodIndexType('A', bodo.types.string_type)`|
| `bodo.RangeIndexType(name_type)`| RangeIndex with a given name type. <br> *e.g.*, `bodo.RangeIndexType(bodo.types.string_type)`|
| `bodo.StringIndexType(name_type)`| Index of strings with a given name type. <br> *e.g.*, `bodo.StringIndexType(bodo.types.string_type)`|
| `bodo.BinaryIndexType(name_type)`| Index of binary values with a given name type. <br> *e.g.*, `bodo.BinaryIndexType(bodo.types.string_type)`|
| `bodo.TimedeltaIndexType(name_type)`| Index of timedelta64 values with a given name type.<br> *e.g.*, `bodo.TimedeltaIndexType(bodo.types.string_type)`|
| `bodo.types.SeriesType(dtype=data_type, index=index_type, name_typ=name_type)`| Series with a given data type, index type, and name type. <br> *e.g.*, `bodo.types.SeriesType(bodo.types.float32, bodo.DatetimeIndexType(bodo.types.string_type), bodo.types.string_type)`|
| `bodo.types.DataFrameType(data_types_tuple, index_type, column_names)`| DataFrame with a tuple of data types, an index type, and the names of the columns. <br> *e.g.*, `bodo.types.DataFrameType((bodo.types.int64[::1], bodo.types.float64[::1]), bodo.RangeIndexType(bodo.types.none), ("A", "B"))`|



## Compile Time Constants {#require_constants}

Unlike regular Python, which is dynamically typed, Bodo needs to be able
to type all functions at compile time. While in most cases, the output
types depend solely on the input types, some APIs require knowing exact
values in order to produce accurate types.

As an example, consider the `iloc` DataFrame API. This API can be used
to selected a subset of rows and columns by passing integers or slices
of integers. A Bodo JIT version of a function calling this API might
look like:

```py
import numpy as np
import pandas as pd
import bodo

@bodo.jit
def df_iloc(df, rows, columns):
   return df.iloc[rows, columns]

df = pd.DataFrame({'A': np.arange(100), 'B': ["A", "B", "C", "D"]* 25})
print(df_iloc(df, slice(1, 4), 0))
```

If we try to run this file, we will get an error message:

```console
$ python iloc_example.py
Traceback (most recent call last):
File "iloc_example.py", line 10, in <module>
   df_iloc(df, slice(1, 4), 0)
File "/my_path/bodo/numba_compat.py", line 1195, in _compile_for_args
   raise error
bodo.utils.typing.BodoError: idx2 in df.iloc[idx1, idx2] should be a constant integer or constant list of integers

File "iloc_example.py", line 7:
def df_iloc(df, rows, columns):
   return df.iloc[rows, columns]
```

The relevant part of the error message is
`idx2 in df.iloc[idx1, idx2] should be a constant integer or constant list of integers`.

This error is thrown because depending on the value of `columns`, Bodo
selects different columns with different types. When `columns=0` Bodo
will need to compile code for numeric values, but when `columns=1` Bodo
needs to compile code for strings, so it cannot properly type this
function.

To resolve this issue, you will need to replace `columns` with a literal
integer. If instead the Bodo function is written as:

```py
import numpy as np
import pandas as pd
import bodo

@bodo.jit
def df_iloc(df, rows):
   return df.iloc[rows, 0]

df = pd.DataFrame({'A': np.arange(100), 'B': ["A", "B", "C", "D"]* 25})
print(df_iloc(df, slice(1, 4)))
```

Bodo now can see that the output DataFrame should have a single `int64`
column and it is able to compile the code.

Whenever a value needs to be known for typing purposes, Bodo will throw
an error that indicates some argument requires `a constant value`. All
of these can be resolved by making this value a literal. Alternatively,
some APIs support other ways of specifying the output types, which will
be indicated in the error message.


## Integer NA issue in Pandas {#integer-na-issue-pandas}

DataFrame and Series objects with integer data need special care due to
[integer NA issues in
Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions){target=blank}.
By default, Pandas dynamically converts integer columns to floating
point when missing values (NAs) are needed (which can result in loss of
precision). This is because Pandas uses the NaN floating point value as
NA, and Numpy does not support NaN values for integers. Bodo does not
perform this conversion unless enough information is available at
compilation time.

Pandas introduced a new [nullable integer data
type](https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html#integer-na){target=blank}
that can solve this issue, which is also supported by Bodo. For example,
this code reads column `A` into a nullable integer array
(the capital `"I"` denotes nullable integer type):

```py
@bodo.jit
def example(fname):
  dtype = {'A': 'Int64', 'B': 'float64'}
  df = pd.read_csv(fname,
      names=dtype.keys(),
      dtype=dtype,
  )
  ...
```


## Type Inference for Object Data

Pandas stores some data types (e.g. strings) as object arrays which are
untyped. Therefore, Bodo needs to infer the actual data type of object
arrays when dataframes or series values are passed to JIT functions from
regular Python. Bodo uses the first non-null value of the array to
determine the type, and throws a warning if the array is empty or all
nulls:

```console
BodoWarning: Empty object array passed to Bodo, which causes ambiguity in typing. This can cause errors in parallel execution.
```

In this case, Bodo assumes the array is a string array which is the most
common. However, this can cause errors if a distributed dataset is
passed to Bodo, and some other processor has non-string data. This
corner case can usually be avoided by load balancing the data across
processors to avoid empty arrays.
