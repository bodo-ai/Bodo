DataFrame
~~~~~~~~~

Bodo provides extensive DataFrame support documented below.

``pd.Dataframe``
*****************

* :class:`pandas.DataFrame` ``(data=None, index=None, columns=None, dtype=None, copy=None)``

    `Supported arguments`:

    .. list-table::
       :widths: 25 35 40
       :header-rows: 1

       * - argument
         - datatypes
         - other requirements
       * - ``data``
         - - constant key dictionary
           - 2D Numpy array
         - - columns argument is required when using a 2D Numpy array
       * - ``index``
         - - List
           - Tuple
           - Pandas index types
           - Pandas array types
           - Pandas series types
           - Numpy array types
         -
       * - ``columns``
         - - Constant list of String
           - Constant tuple of String
         - - **Must be constant at Compile Time**
       * - ``dtype``
         - All values supported with ``dataframe.astype`` (see below)
         -
       * - ``copy``
         - - boolean
         - - **Must be constant at Compile Time**

Attributes and underlying data
******************************

``pd.Dataframe.columns``
^^^^^^^^^^^^^^^^^^^^^^^^^
    * :attr:`pandas.DataFrame.columns`

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
            ...   return df.columns
            >>> f()
            Index(['A', 'B', 'C'], dtype='object')

``pd.Dataframe.dtypes``
^^^^^^^^^^^^^^^^^^^^^^^^^

    * :attr:`pandas.DataFrame.dtypes`

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
            ...   return df.dtypes
            >>> f()
            A              int64
            B             string
            C    timedelta64[ns]
            dtype: object

``pd.Dataframe.empty``
^^^^^^^^^^^^^^^^^^^^^^^^^
    * :attr:`pandas.DataFrame.empty`

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df1 = pd.DataFrame({"A": [1,2,3]})
            ...   df2 = pd.DataFrame()
            ...   return df1.empty, df2.empty
            >>> f()
            (False, True)

``pd.Dataframe.index``
^^^^^^^^^^^^^^^^^^^^^^^^^
    * :attr:`pandas.DataFrame.index`

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3]}, index=["x", "y", "z"])
            ...   return df.index
            >>> f()
            Index(['x', 'y', 'z'], dtype='object')

``pd.Dataframe.ndim``
^^^^^^^^^^^^^^^^^^^^^^^^^
    * :attr:`pandas.DataFrame.ndim`

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
            ...   return df.ndim
            >>> f()
            2

``pd.Dataframe.select_dtypes``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * :meth:`pandas.DataFrame.select_dtypes` ``(include=None, exclude=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``include``
             - - string
               - type
               - List or tuple of string/type
             - - **Must be constant at Compile Time**
           * - ``exclude``
             - - string
               - type
               - List or tuple of string/type
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df= pd.DataFrame({"A": [1], "B": ["X"], "C": [pd.Timedelta(10, unit="D")], "D": [True], "E": [3.1]})
            ...   out_1 = df_l.select_dtypes(exclude=[np.float64, "bool"])
            ...   out_2 = df_l.select_dtypes(include="int")
            ...   out_3 = df_l.select_dtypes(include=np.bool_, exclude=(np.int64, "timedelta64[ns]"))
            ...   formated_out = "\n".join([out_1.to_string(), out_2.to_string(), out_3.to_string()])
            ...   return formated_out
            >>> f()
               A  B       C
            0  1  X 10 days
              A
            0  1
                  D
            0  True

``pd.Dataframe.filter``
^^^^^^^^^^^^^^^^^^^^^^^
    * :meth:`pandas.DataFrame.filter` ``(items=None, like=None, regex=None, axis=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``items``
             - - Constant list of String
           * - ``like``
             - - Constant string
           * - ``regex``
             - - Constant String
           * - ``axis`` (only supports the "column" axis)
             - - Constant String
               - Constant integer


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]})
            ...   filtered_df_1 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(items = ["A"])
            ...   filtered_df_2 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(like ="hello", axis = "columns")
            ...   filtered_df_3 = pd.DataFrame({"ababab": [1], "hello world": [2], "A": [3]}).filter(regex="(ab){3}", axis = 1)
            ...   formated_out = "\n".join([filtered_df_1.to_string(), filtered_df_2.to_string(), filtered_df_3.to_string()])
            ...   return formated_out
            >>> f()
               A
            0  3
              hello world
            0            2
              ababab
            0       1

``pd.Dataframe.shape``
^^^^^^^^^^^^^^^^^^^^^^^
    * :attr:`pandas.DataFrame.shape`

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [3,4,5]})
            ...   return df.shape
            >>> f()
            (3, 2)

``pd.Dataframe.size``
^^^^^^^^^^^^^^^^^^^^^^^
    * :attr:`pandas.DataFrame.size`

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [3,4,5]})
            ...   return df.size
            >>> f()
            6

``pd.Dataframe.to_numpy``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * :meth:`pandas.DataFrame.to_numpy` ``(dtype=None, copy=False, na_value=NoDefault.no_default)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``copy``
             - - boolean

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
            ...   return df.to_numpy()
            >>> f()
            [[1.  3.1]
             [2.  4.2]
             [3.  5.3]]

``pd.Dataframe.values``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * :attr:`pandas.DataFrame.values` (only for numeric dataframes)

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
            ...   return df.values
            >>> f()
            [[1.  3.1]
             [2.  4.2]
             [3.  5.3]]

Conversion
***********


``pd.Dataframe.astype``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * :meth:`pandas.DataFrame.astype` ``(dtype, copy=True, errors='raise')``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``dtype``
             - - dict of string column names keys, and Strings/types values
               - String (string must be parsable by ``np.dtype``)
               - Valid type (see types)
               - The following functions: float, int, bool, str
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [3.1,4.2,5.3]})
            ...   return df.astype({"A": float, "B": "datetime64[ns]"})
            >>> f()
                 A                             B
            0  1.0 1970-01-01 00:00:00.000000003
            1  2.0 1970-01-01 00:00:00.000000004
            2  3.0 1970-01-01 00:00:00.000000005

``pd.Dataframe.copy``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * :meth:`pandas.DataFrame.copy` ``(deep=True)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``copy``
             - - boolean


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3]})
            ...   shallow_df = df.copy(deep=False)
            ...   deep_df = df.copy()
            ...   shallow_df["A"][0] = -1
            ...   formated_out = "\n".join([df.to_string(), shallow_df.to_string(), deep_df.to_string()])
            ...   return formated_out
            >>> f()
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


``pd.Dataframe.isna``
^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.isna` ``()``

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,None,3]})
            ...   return df.isna()
            >>> f()
                   A
            0  False
            1   True
            2  False


``pd.Dataframe.isnull``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * :meth:`pandas.DataFrame.isnull` ``()``

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,None,3]})
            ...   return df.isnull()
            >>> f()
                   A
            0  False
            1   True
            2  False


``pd.Dataframe.notna``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.notna` ``()``

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,None,3]})
            ...   return df.notna()
            >>> f()
                   A
            0   True
            1  False
            2   True

``pd.Dataframe.notnull``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.notnull` ``()``

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,None,3]})
            ...   return df.notnull()
            >>> f()
                   A
            0   True
            1  False
            2   True


``pd.Dataframe.info``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.info` ``(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None)``

        `Supported arguments`: None

        `Example Usage`::

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": ["X", "Y", "Z"], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(10, unit="H"), pd.Timedelta(10, unit="S")]})
            ...   return df.info()
            >>> f()
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

``pd.Dataframe.infer_objects``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :meth:`pandas.DataFrame.infer_objects` ``()``

    `Example Usage`::

        >>> @bodo.jit
        ... def f():
        ...   df = pd.DataFrame({"A": [1,2,3]})
        ...   return df.infer_objects()
           A
        0  1
        1  2
        2  3

    .. note::
      Bodo does not internally use the object dtype, so types are never inferred. As a result, this API just produces a deep copy, consistent with Pandas.

Indexing, iteration
********************

``pd.Dataframe.head``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.head` ``(n=5)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``head``
             - - integer

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   return pd.DataFrame({"A": np.arange(1000)}).head(3)
               A
            0  0
            1  1
            2  2

``pd.Dataframe.iat``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :attr:`pandas.DataFrame.iat`

        We only support indexing using ``iat`` using a pair of integers. We require that the second int
        (the column integer) is a compile time constant


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   df.iat[0, 0] = df.iat[2,2]
            ...   return df
            >>> f()
               A  B  C
            0  9  4  7
            1  2  5  8
            2  3  6  9

``pd.Dataframe.iloc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.iloc`

        `getitem`:

          - ``df.iloc`` supports single integer indexing (returns row as series) ``df.iloc[0]``
          - ``df.iloc`` supports single list/array/series of integers/bool ``df.iloc[[0,1,2]]``
          - for tuples indexing ``df.iloc[row_idx, col_idx]`` we allow:
            - ``row_idx`` to be int list/array/series of integers/bool slice
            - ``col_idx`` to be constant int, constant list of integers, or constant slice
          - e.g.: ``df.iloc[[0,1,2], :]``

        `setitem`:

          - ``df.iloc`` only supports scalar setitem
          - ``df.iloc`` only supports tuple indexing ``df.iloc[row_idx, col_idx]``
          - ``row_idx`` can be anything supported for series setitem:

             - int
             - list/array/series of integers/bool
             - slice
          - ``col_idx`` can be:

              constant int, constant list/tuple of integers



        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   df.iloc[0, 0] = df.iloc[2,2]
            ...   df.iloc[1, [1,2]] = df.iloc[0, 1]
            ...   df["D"] = df.iloc[0]
            ...   return df
            >>> f()
               A  B  C  D
            0  9  4  7  7
            1  2  4  4  4
            2  3  6  9  9

``pd.Dataframe.insert``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.insert` ``(loc, column, value, allow_duplicates=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``loc``
             - - constant integer
           * - ``column``
             - - constant string
           * - ``value``
             - - scalar
               - list/tuple
               - Pandas/Numpy array
               - Pandas index types
               - series
           * - ``allow_duplicates``
             - - constant boolean


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   df.insert(3, "D", [-1,-2,-3])
            ...   return df
            >>> f()
              A  B  C  D
            0  1  4  7 -1
            1  2  5  8 -2
            2  3  6  9 -3


``pd.Dataframe.isin``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.isin` ``(values)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``values``
             - - DataFrame (must have same indices) + iterable type
               - Numpy array types
               - Pandas array types
               - List/Tuple
               - Pandas Index Types (excluding interval Index and MultiIndex)

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   isin_1 = df.isin([1,5,9])
            ...   isin_2 = df.isin(pd.DataFrame({"A": [4,5,6], "C": [7,8,9]}))
            ...   formated_out = "\n".join([isin_1.to_string(), isin_2.to_string()])
            ...   return formated_out
            >>> f()
                  A      B      C
            0  True   False  False
            1  False  True   False
            2  False  False  True
                  A      B     C
            0  False  False  True
            1  False  False  True
            2  False  False  True

        .. note::

            ``DataFrame.isin`` ignores DataFrame indices. For example:

            .. code-block:: ipython3

                >>> @bodo.jit
                ... def f():
                ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
                ...   return df.isin(pd.DataFrame({"A": [1,2,3]}, index=["A", "B", "C"]))
                >>> f()
                        A      B      C
                        0  True  False  False
                        1  True  False  False
                        2  True  False  False

                >>> def f():
                ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
                ...   return df.isin(pd.DataFrame({"A": [1,2,3]}, index=["A", "B", "C"]))
                >>> f()
                        A      B      C
                        0  False  False  False
                        1  False  False  False
                        2  False  False  False


``pd.Dataframe.itertuples``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.itertuples` ``(index=True, name='Pandas')``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   for x in pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}).itertuples():
            ...      print(x)
            ...      print(x[0])
            ...      print(x[2:])
            >>> f()
            Pandas(Index=0, A=1, B=4, C=7)
            0
            (4, 7)
            Pandas(Index=1, A=2, B=5, C=8)
            1
            (5, 8)
            Pandas(Index=2, A=3, B=6, C=9)
            2
            (6, 9)


``pd.Dataframe.query``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.query` ``(expr, inplace=False, **kwargs)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``expr``
             - - Constant String

        `Example Usage`:

        .. code-block:: ipython3

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

``pd.Dataframe.tail``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.tail` ``(n=5)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``n``
             - - Integer

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   return pd.DataFrame({"A": np.arange(1000)}).tail(3)
            >>> f()
                  A
            997  997
            998  998
            999  999

    * :meth:`pandas.DataFrame.where` ``(cond, other=np.nan, inplace=False, axis=1, level=None, errors='raise', try_cast=NoDefault.no_default)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 35
           :header-rows: 1

           * - argument
             - datatypes
             - notes
           * - ``cond``
             - - Boolean DataFrame
               - Boolean Series
               - Boolean Array
             - - If 1-dimensional array or Series is provided, equivalent to Pandas ``df.where`` with ``axis=1``.
           * - ``other``
             - - Scalar
               - DataFrame
               - Series
               - 1 or 2-D Array
               - ``None``
             - - Data types in ``other`` must match corresponding entries in DataFrame.
               - ``None`` or omitting argument defaults to the respective ``NA`` value for each type.

        .. note::
            DataFrame can contain categorical data if ``other`` is a scalar.

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f(df, cond, other):
            ...   return df.where(cond, other)
            >>> df = pd.DataFrame({"A": [1,2,3], "B": [4.3, 2.4, 1.2]})
            >>> cond = df > 2
            >>> other = df + 100
            >>> f(df, cond, other)
                 A      B
            0  101    4.3
            1  102    2.4
            2    3  101.2

    * :meth:`pandas.DataFrame.mask` ``(cond, other=np.nan, inplace=False, axis=1, level=None, errors='raise', try_cast=NoDefault.no_default)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 35
           :header-rows: 1

           * - argument
             - datatypes
             - notes
           * - ``cond``
             - - Boolean DataFrame
               - Boolean Series
               - Boolean Array
             - - If 1-dimensional array or Series is provided, equivalent to Pandas ``df.mask`` with ``axis=1``.
           * - ``other``
             - - Scalar
               - DataFrame
               - Series
               - 1 or 2-D Array
               - ``None``
             - - Data types in ``other`` must match corresponding entries in DataFrame.
               - ``None`` or omitting argument defaults to the respective ``NA`` value for each type.

        .. note::
            DataFrame can contain categorical data if ``other`` is a scalar.

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f(df, cond, other):
            ...   return df.mask(cond, other)
            >>> df = pd.DataFrame({"A": [1,2,3], "B": [4.3, 2.4, 1.2]})
            >>> cond = df > 2
            >>> other = df + 100
            >>> f(df, cond, other)
                A      B
            0    1  104.3
            1    2  102.4
            2  103    1.2

Function application, GroupBy & Window
***************************************

``pd.Dataframe.apply``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.apply` ``(func, axis=0, raw=False, result_type=None, args=(), _bodo_inline=False, **kwargs)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``func``
             - - function (e.g. lambda) (axis must = 1)
               - jit function (axis must = 1)
               - String which refers to a supported DataFrame method
             - - **Must be constant at Compile Time**
           * - ``axis``
             - - Integer (0, 1)
               - String (only if the method takes axis as an argument )
             - - **Must be constant at Compile Time**
           * - ``_bodo_inline``
             - - boolean
             - - **Must be constant at Compile Time**

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.apply(lambda x: x["A"] * (x["B"] + x["C"]))
            >>> f()
            0    11
            1    26
            2    45
            dtype: int64


        .. note ::

            Supports extra ``_bodo_inline`` boolean argument to manually control bodo's inlining behavior.
            Inlining user-defined functions (UDFs) can potentially improve performance at the expense of
            extra compilation time. Bodo uses heuristics to make a decision automatically if ``_bodo_inline`` is not provided.

``pd.Dataframe.groupby``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.groupby` ``(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``by``
             - - String column label
               - List/Tuple of column labels
             - - **Must be constant at Compile Time**
           * - ``as_index``
             - - boolean
             - - **Must be constant at Compile Time**
           * - ``dropna``
             - - boolean
             - - **Must be constant at Compile Time**


        .. note:: ``sort=False`` and ``observed=True`` are set by default. These are the only support values for sort and observed. For more information on using groupby, see :ref:`the groupby Section <pd_groupby_section>`.


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,1,2,2], "B": [-2,-2,2,2]})
            ...   return df.groupby("A").sum()
            >>> f()
               B
            A
            1 -4
            2  4


``pd.Dataframe.rolling``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.rolling` ``(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``window``
             - - Integer
               - String (must be parsable as a time offset)
               - ``datetime.timedelta``
               - ``pd.Timedelta``
               - List/Tuple of column labels
             -
           * - ``min_periods``
             - - Integer
             -
           * - ``center``
             - - boolean
             -
           * - ``on``
             - - Scalar column label
             - - **Must be constant at Compile Time**
           * - ``dropna``
             - - boolean
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3,4,5]})
            ...   return df.rolling(3,center=True).mean()
            >>> f()
                 A
            0  NaN
            1  2.0
            2  3.0
            3  4.0
            4  NaN

        For more information, please see :ref:`the Window section <pd_window_section>`.


Computations / Descriptive Stats
********************************

``pd.Dataframe.abs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.abs` ``()``

        Only supported for dataframes containing numerical data and Timedeltas

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,-2], "B": [3.1,-4.2], "C": [pd.Timedelta(10, unit="D"), pd.Timedelta(-10, unit="D")]})
            ...   return df.abs()
            >>> f()
               A    B       C
            0  1  3.1 10 days
            1  2  4.2 10 days

``pd.Dataframe.corr``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.corr` ``(method='pearson', min_periods=1)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``min_periods``
             - - Integer

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [.9, .8, .7, .4], "B": [-.8, -.9, -.8, -.4], "c": [.7, .7, .7, .4]})
            ...   return df.corr()
            >>> f()
                      A         B        c
            A  1.000000 -0.904656  0.92582
            B -0.904656  1.000000 -0.97714
            c  0.925820 -0.977140  1.00000

``pd.Dataframe.count``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.count` ``(axis=0, level=None, numeric_only=False)``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1, None, 3], "B": [None, 2, None]})
            ...   return df.count()
            >>> f()
            A    2
            B    1


``pd.Dataframe.cov``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.cov` ``(min_periods=None, ddof=1)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``min_periods``
             - - Integer

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [0.695, 0.478, 0.628], "B": [-0.695, -0.478, -0.628], "C": [0.07, -0.68, 0.193]})
            ...   return df.cov()
            >>> f()
                      A         B         C
            A  0.012346 -0.012346  0.047577
            B -0.012346  0.012346 -0.047577
            C  0.047577 -0.047577  0.223293

``pd.Dataframe.cumprod``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.cumprod` ``(axis=None, skipna=True)``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1, 2, 3], "B": [.1,np.NaN,12.3],})
            ...   return df.cumprod()
            >>> f()
               A    B
            0  1  0.1
            1  2  NaN
            2  6  NaN

        .. note::
          Not supported for dataframe with nullable integer.


``pd.Dataframe.cumsum``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.cumsum` ``(axis=None, skipna=True)``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1, 2, 3], "B": [.1,np.NaN,12.3],})
            ...   return df.cumsum()
            >>> f()
               A    B
            0  1  0.1
            1  3  NaN
            2  6  NaN

        .. note::
          Not supported for dataframe with nullable integer.

``pd.Dataframe.describe``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.describe` ``(percentiles=None, include=None, exclude=None, datetime_is_numeric=False)``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [pd.Timestamp(2000, 10, 2), pd.Timestamp(2001, 9, 5), pd.Timestamp(2002, 3, 11)]})
            ...   return df.describe()
            >>> f()
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

``pd.Dataframe.diff``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.diff` ``(periods=1, axis=0)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``periods``
             - - Integer

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [pd.Timestamp(2000, 10, 2), pd.Timestamp(2001, 9, 5), pd.Timestamp(2002, 3, 11)]})
            ...   return df.diff(1)
            >>> f()
                 A        B
            0  NaN      NaT
            1  1.0 338 days
            2  1.0 187 days

        .. note::
          Only supported for dataframes containing float, non-null int, and datetime64ns values


``pd.Dataframe.max``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.max` ``(axis=None, skipna=None, level=None, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.max(axis=1)
            >>> f()
            0    7
            1    8
            2    9

        .. note::
          Only supported for dataframes containing float, non-null int, and datetime64ns values.


``pd.Dataframe.mean``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.mean` ``axis=None, skipna=None, level=None, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.mean(axis=1)
            >>> f()
            0    4.0
            1    5.0
            2    6.0

        .. note::
          Only supported for dataframes containing float, non-null int, and datetime64ns values.


``pd.Dataframe.median``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.median` ``axis=None, skipna=None, level=None, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.median(axis=1)
            >>> f()
            0    4.0
            1    5.0
            2    6.0

        .. note::
          Only supported for dataframes containing float, non-null int, and datetime64ns values.


``pd.Dataframe.min``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.min` ``(axis=None, skipna=None, level=None, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.min(axis=1)
            >>> f()
            0    1
            1    2
            2    3

        .. note::
          Only supported for dataframes containing float, non-null int, and datetime64ns values.

``pd.Dataframe.nunique``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.nunique` ``(axis=0, dropna=True)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``dropna``
             - - boolean

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [1,1,1], "C": [4, None, 6]})
            ...   return df.nunique()
            >>> f()
            A    3
            B    1
            C    2

``pd.Dataframe.pct_change``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.pct_change` ``(periods=1, fill_method='pad', limit=None, freq=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``periods``
             - - Integer


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [10,100,1000,10000]})
            ...   return df.pct_change()
            >>> f()
                A
            0  NaN
            1  9.0
            2  9.0
            3  9.0

``pd.Dataframe.pipe``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.pipe` ``(func, *args, **kwargs)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``func``
             - - JIT function or callable defined within a JIT function.
             - - Additional arguments for ``func`` can be passed as additional arguments.

        .. note::

            ``func`` cannot be a tuple

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   def g(df, axis):
            ...       return df.max(axis)
            ...   df = pd.DataFrame({"A": [10,100,1000,10000]})
            ...   return df.pipe(g, axis=0)
            ...
            >>> f()
            A    10000
            dtype: int64

``pd.Dataframe.prod``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.prod` ``(axis=None, skipna=None, level=None, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.prod(axis=1)
            >>> f()
            A      6
            B    120
            C    504
            dtype: int64

``pd.Dataframe.product``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.product` ``(axis=None, skipna=None, level=None, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.product(axis=1)
            >>> f()
            A      6
            B    120
            C    504
            dtype: int64

``pd.Dataframe.quantile``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.quantile` ``(q=0.5, axis=0, numeric_only=True, interpolation='linear')``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``q``
             - - Float or Int
             - - must be 0<= q <= 1
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.quantile()
            >>> f()
            A    2.0
            B    5.0
            C    8.0
            dtype: float64
            dtype: int64

``pd.Dataframe.std``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.std` ``(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.std(axis=1)
            >>> f()
            0    3.0
            1    3.0
            2    3.0
            dtype: float64


``pd.Dataframe.sum``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.sum` ``(axis=None, skipna=None, level=None, numeric_only=None, min_count=0)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.sum(axis=1)
            >>> f()
            0    12
            1    15
            2    18
            dtype: int64

``pd.Dataframe.var``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.var` ``(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``axis``
             - - Integer (0 or 1)
             - - **Must be constant at Compile Time**


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.var(axis=1)
            >>> f()
            0    9.0
            1    9.0
            2    9.0
            dtype: float64

``pd.Dataframe.memory_usage``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.memory_usage` ``(index=True, deep=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``index``
             - - boolean

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": np.array([1,2,3], dtype=np.int64), "B": np.array([1,2,3], dtype=np.int32), "C": ["1", "2", "3456689"]})
            ...   return df.memory_usage()
            >>> f()
            Index    24
            A        24
            B        12
            C        42
            dtype: int64


Reindexing / Selection / Label manipulation
*******************************************

``pd.Dataframe.drop``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.drop` ``(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')``

        *  Only dropping columns supported, either using ``columns`` argument or setting ``axis=1`` and using the ``labels`` argument
        * ``labels`` and ``columns`` require constant string, or constant list/tuple of string values
        * ``inplace`` supported with a constant boolean value
        * All other arguments are unsupported

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   df.drop(columns = ["B", "C"], inplace=True)
            ...   return df
            >>> f()
               A
            0  1
            1  2
            2  3

``pd.Dataframe.drop_duplicates``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.drop_duplicates` ``(subset=None, keep='first', inplace=False, ignore_index=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``subset``
             - - Constant list/tuple of String column names
               - Constant list/tuple of Integer column names
               - Constant String column names
               - Constant Integer column names

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,1,3,4], "B": [1,1,3,3], "C": [7,8,9,10]})
            ...   return df.drop_duplicates(subset = ["A", "B"])
            >>> f()
               A  B   C
            0  1  1   7
            2  3  3   9
            3  4  3  10

``pd.Dataframe.duplicated``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.duplicated` ``(subset=None, keep='first')``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,1,3,4], "B": [1,1,3,3]})
            ...   return df.duplicated()
            >>> f()
            0    False
            1     True
            2    False
            3    False
            dtype: bool


``pd.Dataframe.first``
^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.first` ``(offset)``

      `Supported arguments`:

      .. list-table::
        :widths: 25 35 40
        :header-rows: 1

        * - argument
          - datatypes
          - other requirements
        * - ``offset``
          - - String or Offset type
          - - String argument be a valid `frequency alias <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_

      .. note::
        DataFrame must have a valid DatetimeIndex and is assumed to already be sorted.
        This function have undefined behavior if the DatetimeIndex is not sorted.

      `Example Usage`:

        .. code-block:: ipython3

          >>> @bodo.jit
          ... def f(df, offset):
          ...     return df.first(offset)
          >>> df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)}, index=pd.date_range(start='1/1/2022', end='12/31/2024', periods=100))
          >>> f(df, "2M")
                                         A    B
          2022-01-01 00:00:00.000000000  0  100
          2022-01-12 01:27:16.363636363  1  101
          2022-01-23 02:54:32.727272727  2  102
          2022-02-03 04:21:49.090909091  3  103
          2022-02-14 05:49:05.454545454  4  104
          2022-02-25 07:16:21.818181818  5  105


``pd.Dataframe.idxmax``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.idxmax` ``(axis=0, skipna=True)``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.idxmax()
            >>> f()
            A    2
            B    2
            C    2
            dtype: int64


``pd.Dataframe.idxmin``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.idxmin` ``(axis=0, skipna=True)``

        `Supported arguments`: None

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.idxmax()
            >>> f()
            A    0
            B    0
            C    20
            dtype: int64


``pd.Dataframe.last``
^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.last` ``(offset)``

      `Supported arguments`:

      .. list-table::
        :widths: 25 35 40
        :header-rows: 1

        * - argument
          - datatypes
          - other requirements
        * - ``offset``
          - - String or Offset type
          - - String argument be a valid `frequency alias <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_

      .. note::
        DataFrame must have a valid DatetimeIndex and is assumed to already be sorted.
        This function have undefined behavior if the DatetimeIndex is not sorted.

      `Example Usage`:

        .. code-block:: ipython3

          >>> @bodo.jit
          ... def f(df, offset):
          ...     return df.last(offset)
          >>> df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100, 200)}, index=pd.date_range(start='1/1/2022', end='12/31/2024', periods=100))
          >>> f(df, "2M")
                                          A    B
          2024-11-05 16:43:38.181818176  94  194
          2024-11-16 18:10:54.545454544  95  195
          2024-11-27 19:38:10.909090912  96  196
          2024-12-08 21:05:27.272727264  97  197
          2024-12-19 22:32:43.636363632  98  198
          2024-12-31 00:00:00.000000000  99  199


``pd.Dataframe.rename``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.rename` ``(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore')``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``mapper``
             - - must be constant dictionary.
             - - Can only be used alongside axis=1
           * - ``columns``
             - - must be constant dictionary
             -
           * - ``axis``
             - - Integer
             - - Can only be used alongside mapper argument
           * - ``copy``
             - boolean
             -
           * - ``inplace``
             - - must be constant boolean
             -

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.rename(columns={"A": "X", "B":"Y", "C":"Z"})
            >>> f()
               X  Y  Z
            0  1  4  7
            1  2  5  8
            2  3  6  9

``pd.Dataframe.reset_index``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.reset_index` ``(level=None, drop=False, inplace=False, col_level=0, col_fill='')``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``level``
             - - Integer
             - - If specified, must drop all levels.
           * - ``drop``
             - - Constant boolean
             -
           * - ``inplace``
             - - Constant boolean
             -

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}, index = ["X", "Y", "Z"])
            ...   return df.reset_index()
            >>> f()
              index  A  B  C
            0     X  1  4  7
            1     Y  2  5  8
            2     Z  3  6  9

``pd.Dataframe.set_index``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.set_index` ``(keys, drop=True, append=False, inplace=False, verify_integrity=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - keys
             - - must be a constant string

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]}, index = ["X", "Y", "Z"])
            ...   return df.set_index("C")
            >>> f()
               A  B
            C
            7  1  4
            8  2  5
            9  3  6


``pd.Dataframe.take``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.take` ``(indices, axis=0, is_copy=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - indices
             - - scalar Integer
               - Pandas Integer Array
               - Numpy Integer Array
               - Integer Series

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.take(pd.Series([-1,-2]))
            >>> f()
               A  B  C
            2  3  6  9
            1  2  5  8

Missing data handling
*********************


``pd.Dataframe.dropna``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.dropna` ``(axis=0, how='any', thresh=None, subset=None, inplace=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``how``
             - - Constant String: either "all" or "any"
           * - ``thresh``
             - - Integer
           * - ``subset``
             - - Constant list/tuple of String column names
               - Constant list/tuple of Integer column names
               - Constant String column names
               - Constant Integer column names

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3,None], "B": [4, 5,None, None], "C": [6, None, None, None]})
            ...   df_1 = df.dropna(how="all", subset=["B", "C"])
            ...   df_2 = df.dropna(thresh=3)
            ...   formated_out = "\n".join([df_1.to_string(), df_2.to_string()])
            ...   return formated_out
            >>> f()
               A  B     C
            0  1  4     6
            1  2  5  <NA>
               A  B  C
            0  1  4  6

``pd.Dataframe.fillna``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.fillna` ``(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``value``
             - - various scalars
             - - Must be of the same type as the filled column
           * - ``inplace``
             - - Constant boolean
             - - ``inplace`` is not supported alongside method
           * - ``method``
             - - One of ``bfill``, ``backfill``, ``ffill`` , or ``pad``
             - - **Must be constant at Compile Time**
               - ``inplace`` is not supported alongside method

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3,None], "B": [4, 5,None, None], "C": [6, None, None, None]})
            ...   return df.fillna(-1)
            >>> f()

``pd.Dataframe.replace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.replace` ``(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``to_replace``
             - - various scalars
             - - Required argument
           * - ``value``
             - - various scalars
             - - Must be of the same type as to_replace

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.replace(1, -1)
            >>> f()
               A  B  C
            0 -1  4  7
            1  2  5  8
            2  3  6  9

Reshaping, sorting, transposing
*******************************


``pd.Dataframe.pivot``
^^^^^^^^^^^^^^^^^^^^^^


    * :meth:`pandas.DataFrame.pivot` ``(values=None, index=None, columns=None)``


        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``values``
             - - Constant Column Label or list of labels
           * - ``index``
             - - Constant Column Label or list of labels
           * - ``columns``
             - - Constant Column Label


        .. note::
          The the number of columns and names of the output DataFrame won't be known
          at compile time. To update typing information on DataFrame you should pass it back to Python.


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": ["X","X","X","X","Y","Y"], "B": [1,2,3,4,5,6], "C": [10,11,12,20,21,22]})
            ...   pivoted_tbl = df.pivot(columns="A", index="B", values="C")
            ...   return pivoted_tbl
            >>> f()
            A     X     Y
            B
            1  10.0   NaN
            2  11.0   NaN
            3  12.0   NaN
            4  20.0   NaN
            5   NaN  21.0
            6   NaN  22.0


``pd.Dataframe.pivot_table``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.pivot_table` ``(values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False, sort=True)``


        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``values``
             - - Constant Column Label or list of labels
           * - ``index``
             - - Constant Column Label or list of labels
           * - ``columns``
             - - Constant Column Label
           * - ``aggfunc``
             - - String Constant


        .. note::
          This code takes two different paths depending on if pivot values are annotated. When
          pivot values are annotated then output columns are set to the annotated values.
          For example, ``@bodo.jit(pivots={'pt': ['small', 'large']})``
          declares the output pivot table ``pt`` will have columns called ``small`` and ``large``.

          If pivot values are not annotated, then the number of columns and names of the output DataFrame won't be known
          at compile time. To update typing information on DataFrame you should pass it back to Python.


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit(pivots={'pivoted_tbl': ['X', 'Y']})
            ... def f():
            ...   df = pd.DataFrame({"A": ["X","X","X","X","Y","Y"], "B": [1,2,3,4,5,6], "C": [10,11,12,20,21,22]})
            ...   pivoted_tbl = df.pivot_table(columns="A", index="B", values="C", aggfunc="mean")
            ...   return pivoted_tbl
            >>> f()
                  X     Y
            B
            1  10.0   NaN
            2  11.0   NaN
            3  12.0   NaN
            4  20.0   NaN
            5   NaN  21.0
            6   NaN  22.0


``pd.Dataframe.sample``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.sample` ``(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None, ignore_index=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``n``
             - - Integer
           * - ``frac``
             - - Float
           * - ``replace``
             - - boolean


        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6], "C": [7,8,9]})
            ...   return df.sample(1)
            >>> f()
               A  B  C
            2  3  6  9

``pd.Dataframe.sort_index``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.sort_index` ``(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``ascending``
             - - boolean
           * - ``na_position``
             - - constant String ("first" or "last")



        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3]}, index=[1,None,3])
            ...   return df.sort_index(ascending=False, na_position="last")
            >>> f()
                 A
            3    3
            1    1
            NaN  2

``pd.Dataframe.sort_values``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.sort_values` ``(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35 40
           :header-rows: 1

           * - argument
             - datatypes
             - other requirements
           * - ``by``
             - - constant String or constant list of strings
             -
           * - ``ascending``
             - - boolean
               - list/tuple of boolean, with length equal to the number of key columns
             -
           * - ``inplace``
             - - Constant boolean
             -
           * - ``na_position``
             - - constant String ("first" or "last")
               - constant list/tuple of String, with length equal to the number of key columns
             -

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,2,None], "B": [4, 5, 6, None]})
            ...   df.sort_values(by=["A", "B"], ascending=[True, False], na_position=["first", "last"], inplace=True)
            ...   return df
            >>> f()
                  A     B
            3  <NA>  <NA>
            0     1     4
            2     2     6
            1     2     5

``pd.Dataframe.to_string``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    *  :meth:`pandas.DataFrame.to_string` ``(buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, min_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None, max_colwidth=None, encoding=None)``

        `Supported arguments`:

          * ``buf``
          * ``columns``
          * ``col_space``
          * ``header``
          * ``index``
          * ``na_rep``
          * ``formatters``
          * ``float_format``
          * ``sparsify``
          * ``index_names``
          * ``justify``
          * ``max_rows``
          * ``min_rows``
          * ``max_cols``
          * ``how_dimensions``
          * ``decimal``
          * ``line_width``
          * ``max_colwidth``
          * ``encoding``



        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3]})
            ...   return df.to_string()
            >>> f()
               A
            0  1
            1  2
            2  3

        .. note::
           * This function is not optimized.
           * When called on a distributed dataframe, the string returned for each rank will be reflective of the dataframe for that rank.

Combining / joining / merging
******************************

``pd.Dataframe.append``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.append` ``(other, ignore_index=False, verify_integrity=False, sort=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``other``
             - - Dataframe
               - list/tuple of Dataframe
           * - ``ignore_index``
             - - constant boolean

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
            ...   return df.append(pd.DataFrame({"A": [-1,-2,-3], "C": [4,5,6]}))
            >>> f()
               A    B    C
            0  1  4.0  NaN
            1  2  5.0  NaN
            2  3  6.0  NaN
            0 -1  NaN  4.0
            1 -2  NaN  5.0
            2 -3  NaN  6.0

``pd.Dataframe.assign``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.assign` ``(**kwargs)``

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
            ...   df2 = df.assign(C = 2 * df["B"], D = lambda x: x.C * -1)
            ...   return df2
            >>> f()
               A  B   C   D
            0  1  4   8  -8
            1  2  5  10 -10
            2  3  6  12 -12


        .. note::
            arguments can be JIT functions, lambda functions, or values that can be used to initialize a Pandas Series.


``pd.Dataframe.join``
^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.join` ``(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``other``
             - - Dataframe
           * - ``on``
             - - constant string column name
               - constant list/tuple of column names

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
            ...   return df.join(on = "A", other=pd.DataFrame({"C": [-1,-2,-3], "D": [4,5,6]}))
            >>> f()
               A  B     C     D
            0  1  4    -2     5
            1  1  5    -2     5
            2  3  6  <NA>  <NA>


        .. note::
           Joined dataframes cannot have common columns. The output dataframe is not sorted by default for better parallel performance


``pd.Dataframe.merge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.merge` ``(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)``

        See :ref:`pd.merge <pd_merge_fn>` for full list of support arguments, and more examples.

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
            ...   return df.merge(pd.DataFrame({"C": [-1,-2,-3], "D": [4,4,6]}), left_on = "B", right_on = "D")
            >>> f()
               A  B  C  D
            0  1  4 -1  4
            1  1  4 -2  4
            2  3  6 -3  6


Time series-related
********************

``pd.Dataframe.shift``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.shift` ``(periods=1, freq=None, axis=0, fill_value=NoDefault.no_default)``

        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``periods``
             - - Integer

        `Example Usage`:

        .. code-block:: ipython3

            >>> @bodo.jit
            ... def f():
            ...   df = pd.DataFrame({"A": [1,1,3], "B": [4,5,6]})
            ...   return df.shift(1)
            >>> f()
                 A    B
            0  NaN  NaN
            1  1.0  4.0
            2  1.0  5.0

        .. note::
          Only supported for dataframes containing numeric, boolean, datetime.date and string types.



.. _pandas-f-out:

Serialization / IO / conversion
*******************************

Also see :ref:`S3` and :ref:`HDFS` configuration requirements and more on :ref:`file_io`.

``pd.Dataframe.to_csv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.to_csv`

      * ``compression`` argument defaults to ``None`` in JIT code. This is the only supported value of this argument.
      * ``mode`` argument supports only the default value ``"w"``.
      * ``errors`` argument supports only the default value ``strict``.
      * ``storage_options`` argument supports only the default value ``None``.
    * :meth:`pandas.DataFrame.to_json`
    * :meth:`pandas.DataFrame.to_parquet`
    * :meth:`pandas.DataFrame.to_sql`

      * See :ref:`Example Usage and more system specific instructions <sql-section>`.
      * Argument ``con`` is supported but only as a string form. SQLalchemy `connectable` is not supported.
      * Argument ``name``, ``schema``, ``if_exists``, ``index``, ``index_label``, ``dtype``, ``method`` are supported.
      * Argument ``chunksize`` is not supported.

Plotting
********

``pd.Dataframe.plot``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * :meth:`pandas.DataFrame.plot` ``(x=None, y=None, kind="line", figsize=None, xlabel=None, ylabel=None, title=None, legend=True, fontsize=None, xticks=None, yticks=None, ax=None)``


        `Supported arguments`:

        .. list-table::
           :widths: 25 35
           :header-rows: 1

           * - argument
             - datatypes
           * - ``x``
             - - Constant String column name
               - Constant integer
           * - ``y``
             - - Constant String column name
               - Constant integer
           * - ``kind``
             - - constant String ("line" or "scatter")
           * - ``figsize``
             - - constant numeric tuple (width, height)
           * - ``xlabel``
             - - constant String
           * - ``ylabel``
             - - constant String
           * - ``title``
             - - constant String
           * - ``legend``
             - - boolean
           * - ``fontsize``
             - - integer
           * - ``xticks``
             - - Constant Tuple
           * - ``yticks``
             - - Constant Tuple
           * - ``ax``
             - - Matplotlib Axes Object
