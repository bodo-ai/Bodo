.. _March_2021:

Bodo 2021.3 Release (Date: 3/25/2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release includes many new features, bug fixes and usability improvements.
Overall, 148 code patches were merged since the last release.

New Features and Improvements
-----------------------------

- Bodo is updated to use Numba 0.53 (latest) and support Python 3.9

- Many improvements to error checking and reporting

- Compilation time is reduced, especially for user-defined functions (UDFs)

- Reduced initialization time when importing Bodo

- Distributed diagnostics improvements:

    - Show distributed diagnostics when raising errors for distributed flag
    - Only show user defined variables in diagnostics level one

- Performance optimizations:

    - Faster groupby ``nunique`` with improved scaling
    - Faster ``setitem`` for categorical arrays

- Connectors:

    - Google Cloud Storage (GCS) support with Parquet
    - Support reading Delta Lake tables
    - Improved Snowflake support
    - Removed s3fs dependency (Bodo now fully relies on Apache Arrow for S3 connectivity)

- Change default parallelism semantics of ``unique()`` to replicated output to match user expectations better

- Support ``objmode`` in groupby apply UDFs

- Pandas coverage:

    - Support ``pd.DataFrame.duplicated()`` with categorical data
    - Groupby support for min/max on categorical data
    - Support for categorical in ``pd.Series.dropna``
    - Support nullable int array in ``pd.Categorical`` constructor
    - Support for ``pd.Series.where`` and ``pd.Series.mask`` with categorical data and a scalar value.
    - Support for ``pd.Series.diff()``
    - Support for ``pd.DataFrame.diff()``
    - Support for ``pd.Series.repeat()``
    - Support list of functions in ``groupby.agg()``
    - Support tuple of UDFs inside ``groupby.agg()`` dictionary case
    - Support single row and scalar UDF output in ``groupby.apply()``
    - Support Categorical values in ``Groupby.shift``
    - Support ``case=False`` in ``Series.str.contains``
    - Support ``mapper`` with ``axis=1`` for ``pd.DataFrame.rename``.
    - Support ``Timedelta64`` data in ``pd.Groupby``
    - Support for ``datetime.date`` arrays in ``Series.max`` and ``Series.min``
    - Support for ``pd.timedelta_range``
    - Support equality between ``datetime64``/``pd.Timestamp`` and ``timedelta64``/``pd.Timedelta``
    - Support for iterating across most index types
    - Support getting the ``name`` attribute of data inside ``df.apply``
    - Support ``Series.reset_index(drop=False)`` for common cases
    - Support ``==`` and ``!=`` on Dataframe and a scalar with a different type
    - Sequential support for ``pd.Series.idxmax``, ``pd.Series.idxmin``,
        ``pd.DataFrame.idxmax``, and ``pd.DataFrame.idxmin`` with Nullable
        and Categorical arrays.

- Python coverage:

    - Support ``datetime.date.replace()``
    - Improved support for ``datetime.date.strftime()``
    - Support for ``calendar.month_abbr``


- SciPy:

    - Initial support for ``scipy.sparse.csr_matrix``


- Scikit-learn:

    - Support for ``sklearn.feature_extraction.text.HashingVectorizer``
