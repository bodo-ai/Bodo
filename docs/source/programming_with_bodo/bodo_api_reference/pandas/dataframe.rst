
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

