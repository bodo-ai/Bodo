

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
