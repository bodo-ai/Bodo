.. _January_2022:

Bodo 2022.1 Release (Date: 1/31/2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release includes many new features and usability improvements.
Overall, 71 code patches were merged since the last release.


New Features and Improvements
-----------------------------

- Bodo is now available with ``pip`` on both Linux and Windows

- Bodo is upgraded to use Numba 0.55.0 (the latest release)

- Bodo can now evaluate JIT functions at compilation time if possible to extract constant values.
  This improves user experience by simplifying type stability requirements. For example, the
  function below can be refactored to be type stable easily:

    .. code-block:: ipython3

      @bodo.jit
      def f(df):
          df.columns = [f"m_{c}" if c not in ["B", "C"] else c for c in df.columns]
          return df

    .. code-block:: ipython3

      @bodo.jit
      def g(df_cols):
          return [f"m_{c}" if c not in ["B", "C"] else c for c in df_cols]

      @bodo.jit
      def f(df):
          df.columns = g(df.columns)
          return df

- Connectors:

  - ``read_csv`` now skips hidden files when reading from a directory.

  - ``read_parquet`` now supports reading a list of files.

  - Improved error handling for both ``read_csv`` and ``read_sql``


- Improved null value handling in user-defined-functions that aren't inlined.

- Truncated error messages with DataFrames with large numbers of columns to improve readability.

- Improved support for the ``logging`` standard library:

  - Support regular ``logging.Logger`` in addition to the ``logging.RootLogger``.

  - Supports passing a logger as a constant global.

  - Supports the attributes: ``level``, ``name``, ``propagate``, ``disabled``, and ``parent``.

  - Supports the methods: ``debug``, ``warning``, ``warn``, ``error``, ``exception``,
    ``critical``, ``log``, and ``setLevel``.

- Improvments to global value handling of the compiler to avoid memory leaks in corner cases.

- Pandas:

  - Support for ``DataFrame.pivot()`` and ``DataFrame.pivot_table()`` without requiring a constant
    list of output columns. Bodo currently only supports limited operations on output DataFrames of pivot,
    so users are recommended to immediately return these DataFrames to Python before doing any further processing
    in Bodo.

  - Support for ``Index.rename``

  - Support for ``Index.is_monotonic``, ``Index.is_montonic_increasing``, and ``Index.is_monotonic_decreasing``

  - Support for ``Index.notna`` and ``Index.notnull``

  - Support for ``Index.drop_duplicates``

  - Support for ``groupby.min``, ``groupby.max``, ``groupby.first``, and ``groupby.last``
    on DataFrames with Categorical columns

  - Support for column slice assignment with ``df.iloc`` (e.g. ``df.iloc[0,:] = 0``)

  - Support for ``Series.first``, ``Series.last``, ``DataFrame.first``, and ``DataFrame.last``
