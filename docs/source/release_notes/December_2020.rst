.. _December_2020:

Bodo 2020.12 Release (Date: 12/30/2020)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release includes many new features, bug fixes and performance improvements.
Overall, 60 code patches were merged since the last release.

New Features and Improvements
-----------------------------

- Bodo is updated to use Numba 0.52 (latest)

- Support for reading CSV and Parquet from Azure Data Lake Storage (ADLS)

- Improved support for UDFs
   - More robust user function handling
   - Improved support for date/time data types in UDFs

- Improved support for rolling window functions
    - Support `raw` argument of ``apply()``
    - Support column selection from rolling objects
    - Support for nullable int values

- Pandas coverage:
    - Support for ``groupby.apply``
    - Support for groupby rolling functions
    - Improved support for dataframe indexing using df.loc/iloc
    - Improve dtype handling in ``read_csv``
    - Support for ``Series.mask``
    - Improved robustness for highly skewed string data (e.g. most of string data is on a few processes due to uneven data distribution)
    - Support for dataframes with repeated column names
    - Support for ``datetime.date`` arrays as Index in ``pivot_table`` and as argument to ``pd.DatetimeIndex``
    - Improved error checking in Pandas implementations
    - Unroll constant loops for type stability in more cases

- Numpy coverage:
   - Support for ``np.hstack``

- Scikit-learn:
   - Support for ``sklearn.preprocessing.StandardScaler`` inside jit functions.
