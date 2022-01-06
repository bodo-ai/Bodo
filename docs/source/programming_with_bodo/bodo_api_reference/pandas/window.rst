.. _pd_window_section:

Window
~~~~~~

Rolling functionality is documented in `pandas.DataFrame.rolling <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html>`_.

* :meth:`pandas.core.window.rolling.Rolling.count` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5], "B": [6,7,None,9,10]})
      ...   return df.rolling(3).count()
          A    B
      0  1.0  1.0
      1  2.0  2.0
      2  3.0  3.0
      3  3.0  2.0
      4  3.0  2.0
      5  3.0  2.0
      6  3.0  3.0


* :meth:`pandas.core.window.rolling.Rolling.sum` ``(engine=None, engine_kwargs=None)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
      ...   return df.rolling(3).sum()
            A     B
      0   NaN   NaN
      1   NaN   NaN
      2   6.0  27.0
      3   9.0   NaN
      4  12.0   NaN
      5  15.0   NaN
      6  18.0  36.0

* :meth:`pandas.core.window.rolling.Rolling.mean` ``(engine=None, engine_kwargs=None)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
      ...   return df.rolling(3).mean()
          A     B
      0  NaN   NaN
      1  NaN   NaN
      2  2.0   9.0
      3  3.0   NaN
      4  4.0   NaN
      5  5.0   NaN
      6  6.0  12.0

* :meth:`pandas.core.window.rolling.Rolling.median` ``(engine=None, engine_kwargs=None)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
      ...   return df.rolling(3).median()
          A     B
      0  NaN   NaN
      1  NaN   NaN
      2  2.0   9.0
      3  3.0   NaN
      4  4.0   NaN
      5  5.0   NaN
      6  6.0  12.0

* :meth:`pandas.core.window.rolling.Rolling.var` ``(ddof=1)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
      ...   return df.rolling(3).var()
          A    B
      0  NaN  NaN
      1  NaN  NaN
      2  1.0  1.0
      3  1.0  NaN
      4  1.0  NaN
      5  1.0  NaN
      6  1.0  1.0


* :meth:`pandas.core.window.rolling.Rolling.std` ``(ddof=1)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
      ...   return df.rolling(3).std()
          A    B
      0  NaN  NaN
      1  NaN  NaN
      2  1.0  1.0
      3  1.0  NaN
      4  1.0  NaN
      5  1.0  NaN
      6  1.0  1.0

* :meth:`pandas.core.window.rolling.Rolling.min` ``(engine=None, engine_kwargs=None)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
      ...   return df.rolling(3).min()
          A     B
      0  NaN   NaN
      1  NaN   NaN
      2  1.0   8.0
      3  2.0   NaN
      4  3.0   NaN
      5  4.0   NaN
      6  5.0  11.0



* :meth:`pandas.core.window.rolling.Rolling.max` ``(engine=None, engine_kwargs=None)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,5,6,7], "B": [8,9,10,None,11,12,13]})
      ...   return df.rolling(3).max()
          A     B
      0  NaN   NaN
      1  NaN   NaN
      2  3.0  10.0
      3  4.0   NaN
      4  5.0   NaN
      5  6.0   NaN
      6  7.0  13.0

* :meth:`pandas.core.window.rolling.Rolling.corr` ``(other=None, pairwise=None, ddof=1)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``other`` (required)
      - DataFrame or Series (cannot contain nullable Integer Types)
      - If called with a DataFrame, ``other`` must be a DataFrame. If called with a Series, ``other`` must be a Series.


  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df1 = pd.DataFrame({"A": [1,2,3,4,5,6,7]})
      ...   df2 = pd.DataFrame({"A": [1,2,3,4,-5,-6,-7]})
      ...   return df1.rolling(3).corr(df2)
                A
      0       NaN
      1       NaN
      2  1.000000
      3  1.000000
      4 -0.810885
      5 -0.907841
      6 -1.000000

* :meth:`pandas.core.window.rolling.Rolling.cov` ``(other=None, pairwise=None, ddof=1)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``other`` (required)
      - DataFrame or Series (cannot contain nullable Integer Types)
      - If called with a DataFrame, ``other`` must be a DataFrame. If called with a Series, ``other`` must be a Series.


  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df1 = pd.DataFrame({"A": [1,2,3,4,5,6,7]})
      ...   df2 = pd.DataFrame({"A": [1,2,3,4,-5,-6,-7]})
      ...   return df1.rolling(3).cov(df2)
          A
      0  NaN
      1  NaN
      2  1.0
      3  1.0
      4 -4.0
      5 -5.0
      6 -1.0


* :meth:`pandas.core.window.rolling.Rolling.apply` ``(func, raw=False, engine=None, engine_kwargs=None, args=None, kwargs=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 25 35
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``func``
      - JIT function or callable defined within a JIT function
      - **Must be constant at Compile Time**
    * - ``raw``
      - boolean
      - **Must be constant at Compile Time**


  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(I):
      ...   df = pd.DataFrame({"A": [1,2,3,4,-5,-6,-7]})
      ...   return df.rolling(3).apply(lambda x: True if x.sum() > 0 else False)
          A
      0  NaN
      1  NaN
      2  1.0
      3  1.0
      4  1.0
      5  0.0
      6  0.0




