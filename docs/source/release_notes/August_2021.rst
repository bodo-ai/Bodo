.. _August_2021:

Bodo 2021.8 Release (Date: 8/30/2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release includes many new features, optimizations, bug fixes and usability improvements.
Overall, 74 code patches were merged since the last release.


New Features and Improvements
-----------------------------

- Bodo is updated to use pandas 1.3 and Arrow 5.0 (latest)

- Updated ``bodo.jit`` flag handling to remove the need for the ``distributed`` flag in most cases.
  Arguments and return types are now automatically inferred as distributed in many cases.
  Automatic inference can be disabled using the new ``returns_maybe_distributed`` and ``args_maybe_distributed``
  flags if necessary.

- Connectors:

    - Improved ``pd.read_sql`` performance on Snowflake by using
      the Snowflake connector APIs directly.
    - Improved performance of ``pd.read_parquet`` when reading large partitioned
      datasets

- Performance improvements:

    - Reduced compilation time for some DataFrame operations
    - General performance improvements in Bodo's execution engine resulting in better
      speed and memory efficiency for a wide range of operations
    - Improved performance of ``merge`` and ``join`` operations
    - Improved performance and scalability of ``groupby`` operations
    - Improved performance of ``groupby.apply``
    - Improved performance of ``groupby.transform``
    - Significantly optimized ``Series.str.contains(..., regex=True)``
    - Improved performance of filtering operations involving string arrays

- Pandas:

    - Support for passing string function names to ``Series.apply``.
      The string can refer to a Series method or a Numpy ufunc.
    - Support for passing string function names to ``DataFrame.apply``. The string can refer to a
      DataFrame method. ``axis`` can be provided if the method takes an ``axis`` argument.
    - Enhanced support for binary arrays, including within series/dataframes
    - ``astype()`` support for casting strings to nullable integers
    - Support for ``operator.mul`` between a ``Timedelta`` scalar and integers Series
    - Support for ``std`` in ``groupby.transform``.

- Scikit-learn:

    - Support for ``sklearn.feature_extraction.text.CountVectorizer``
    - Support for ``coef_`` attribute for ``Ridge``



BodoSQL 2021.8beta Release (Date: 8/30/2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This release adds more SQL coverage, introduces new BodoSQL specific features,
and fixes bugs. Overall, 53 code patches were merged since the last release.


New Features and Improvements
-----------------------------

- Parameterized Queries:

  Parameterized queries allow replacing scalars with Python variables during runtime execution.
  This enables caching more complex BodoSQL queries When paired with Bodo jit: a query parameter can
  change without the need to recompile the query.
  More information and example usage can be found in our documentation.

- SQL Coverage:

  This release added the following additional SQL coverage to BodoSQL. Please
  refer to our documentation for more details regarding usage.

    - Support for != and <=> operators

    - Support for CAST

    - Support for LEAST

    - Support for [NOT] IN with lists of literals

    - Support for the offset optional argument in queries with LIMIT
      (i.e. SELECT A from table LIMIT 1, 4)

    - Initial support for YEAR/MONTH interval literals/scalars.
      Currently these are only supported with addition and subtraction
      operators and cannot be used as a column type.

    - Support the following string functions:

        - CHAR (to convert a value to a string)

        - LENGTH

    - Support for the following Timestamp functions:

        - ADDDATE

        - SUBDATE

        - TIMESTAMPDIFF

        - WEEKDAY

        - YEARWEEK

        - LAST_DAY
