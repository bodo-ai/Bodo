.. _December_2021:

Bodo 2021.12 Release (Date: 12/29/2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release includes many new features and usability improvements.
Overall, 67 code patches were merged since the last release.


New Features and Improvements
-----------------------------

- Significantly upgrades to the Bodo documentation to improve the developer experience

- Improvements to documentation and unsupported attribute handling for Pandas APIs

- Significant enhancements to objmode user experience and robustness, such as automatic output data type checking and automatic conversion if possible

- Improved support for ``re`` package, such as support for ``re`` flags, better support for returning ``None`` when necessary, and better catching of unsupported corner cases

- Support caching functions that take a string as input and create a file path using concatenation. For example:

  .. code-block:: ipython3

    @bodo.jit(cache=True)
    def f(folder):
      return pd.read_parquet(folder + "/example.pq")

- Connectors:

  - Improved ``read_parquet`` runtime performance when reading from S3

  - Decreased compilation time for ``read_csv`` on DataFrames with large number of columns (> 100)


- Improved compilation time for dataframes with large number of columns (>10,000)

- Improved NA handling in User Defined Functions with df.apply when functions are not inlined

- Support for using ``logging.RootLogger.info`` when passing the logger as an argument to a JIT function

- Support for ``datetime.datetime.today``

- Simpler ``bodo.scatterv`` usage from regular Python. Other ranks are ignored but not required to have ``None``
  as their data

- Improved support for map arrays in various operations

- Support ``feature_importances_`` of XGBoost

- Support ``predict_proba`` and ``predict_log_proba`` in Scikit-learn classifier algorithms


- Pandas:

  - Support for Bodo specific argument ``_bodo_upcast_to_float64`` in pd.read_csv. This can be used
    when all data is numeric but schema inference cannot accurate predict data types.

  - Support for using ``DataFrame.to_parquet`` with "wide" DataFrames with large number of columns

  - Support for storing a ``DateTimeIndex`` with ``DataFrame.to_parquet``

  - Support for the 'method' argument in ``DataFrame.fillna`` and ``Series.fillna``

  - Support for ``Series.bfill``, ``Series.ffill``, ``Series.pad``, and ``Series.backfill``

  - Support for ``Series.keys``

  - Support for ``Series.infer_objects`` and ``DataFrame.infer_objects``

  - Decreased runtime when calling ``.astype("categorical")`` on Series with large numbers of categories
