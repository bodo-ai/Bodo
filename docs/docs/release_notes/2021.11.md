Bodo 2021.11 Release (Date: 11/30/2021) {#November_2021}
========================================

This release includes many new features, optimizations, bug fixes and
usability improvements. Overall, 107 code patches were merged since the
last release.

## New Features and Improvements

-   Support for "wide" DataFrames with large number of columns:
    -   Bodo compiler is transitioning to a new internal dataframe
        compilation format that substantially decreases compliation time
        for dataframes with thousands of columns.

        All DataFrame APIs will transition to this new format over time.

    -   `read_csv`, `read_parquet`, `bodo.gatherv` and dataframe
        filtering are upgraded to support this new format in this
        release.
-   Connectors:
    -   Significantly improved performance when reading Parquet from S3
        (up to 10x faster read depending on the dataset).
    -   General support for predicate pushdown when reading from Parquet
        (filtering rows at the storage level).
    -   Improvements to BodoSQL's filter pushdown, such as higher
        compiler accuracy in detecting possible filters.
    -   Faster `read_parquet` compilation time by validating the schema
        only at runtime.
    -   Faster `pd.read_csv()` execution time with large numbers of
        columns.
-   Bodo automatically maintains type information when passing
    DataFrames and Series between Bodo and regular Python. This avoids
    potential typing issues when parallel data chunks do not have enough
    non-null data for automatic type inference.
-   Improved error messages and documentation.
-   Pandas:
    -   Support for Array of dictionary outputs of `DataFrame.apply()`
        and `Series.apply()`
    -   Support for Array of dictionary inputs to `pd.concat()`
    -   Support for `Series.astype(str)` with Categorical type for
        non-string categories.
    -   Support for callable arguments to `DataFrame.assign()`
    -   Support for passing a list as `skiprows` of `pd.read_csv()`
    -   Support for `low_memory` argument in `pd.read_csv()`
    -   Support for using a string label for indexing Series with string
        Index (for non-parallel Series)
    -   Support for initializing a Series with a constant dictionary
    -   Support for `subset` argument to `DataFrame.drop_duplicates`
    -   Support for `DataFrame.plot` with arguments `x`, `y`, `kind`,
        `figsize`, `xlabel`, `ylabel`, `title`, `legend`, `fontsize`,
        `xticks`, `yticks`, and `ax`. `DataFrame.plot` behaves the same
        as Bodo's Matplotlib support.
    -   Support for `DataFrame.groupby.head`
-   Numpy:
    -   Support for `np.select`
-   ML:
    -   Support `predict_proba` and `predict_log_proba` for
        `RandomForestClassifier`, `SGD Classifier` and
        `LogisticRegression`
    -   Support `predict_proba` for XGBoostClassifier
    -   Support for `sklearn.metrics.confusion_matrix`

BodoSQL 2021.11beta Release (Date: 11/30/2021)

This release includes SQL bug fixes and support for Bodo's filter
pushdown from BodoSQL. Most of the improvements to BodoSQL are
integrating enhancements made to Bodo. Overall, 10 code patches were
merged since the last release.

## New Features and Improvements

-   Support for a new filepath API `bodosql.TablePath`. This API takes
    the path and file type and uses this to load/remove the data within
    the query.

    For example:

        bc = bodosql.BodoSQLContext("table1": bodosql.TablePath("myfile.pq", "parquet"))
        return bc.query("Select A from table1")

    This is functionally equivalent to using the Pandas
    `read_` functions inside a Bodo function, but it may
    have some additional performance optimizations.

    Currently only Parquet files are supported.

-   Support for Bodo's filter pushdown when using the
    `bodosql.TablePath` API.

-   Reduced compliation and execution time when using the `FIRST_VALUE`
    function repeatedly on the same exact window.

-   SQL Coverage

    This release added the following additional SQL coverage to BodoSQL.
    Please refer to our
    [documentation](https://docs.bodo.ai/latest/source/programming_with_bodo/BodoSQL.html#supported-operations)
    for more details regarding usage.

    -   Support for omitting the second argument from the `ROUND`
        function (defaults to 0).
    -   Support for providing an integer as the second argument
        `DATE_ADD` and `DATE_SUB`. If you pass
        an integer, it is assigned `days` as its unit.
