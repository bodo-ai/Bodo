
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
