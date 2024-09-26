Bodo 2021.4 Release (Date: 4/19/2021) {#April_2021}
=====================================

This release includes many new features, bug fixes and usability
improvements. Overall, 98 code patches were merged since the last
release.

## New Features and Improvements

-   Bodo is available for Windows as a Conda package (similar to Linux
    and macOS)

-   Removed boost library dependency

-   Many improvements to error checking and reporting, including:

    -   Internal compiler errors and stack traces are now avoided more
        effectively (clear errors are thrown)
    -   Ensure that an error is thrown if user specifies an argument
        as distributed but it must be replicated
    -   Improvements in error checking for user-defined functions
        (UDFs)

-   Connectors:

    -   Support for writing partitioned Parquet datasets
        (`df.to_parquet` with `partition_cols` parameter)
    -   Support for S3 anonymous access with
        `storage_options={"anon": True}` in `pd.read_parquet()`
    -   Parquet read: optimized metadata collection for nested parquet
        directories (includes hive-partitioned dataset)
    -   To reduce Parquet read time, schema validation of multi-file
        parquet datasets can be disabled with
        `bodo.parquet_validate_schema=False`

-   Reduced compilation time for Pandas APIs

-   Improved compilation time for `df.head/tail`

-   Support for format spec in f-strings, for example: `f"{a:0.0%}"`

-   Support for arrays in `bodo.rebalance()`

-   Pandas coverage:

    -   Support for `df.filter` for filtering columns
    -   Support for `indicator=True` in `pd.merge()`
    -   Support for `DataFrame/Series/GroupBy.pipe()`
    -   Support for setting dataframe columns using a 2D array
    -   Support for string and nullable arrays (e.g. pd.Int64Dtype) in
        `DataFrame/Series.shift()`
    -   Support for `pandas.tseries.offsets.MonthBegin`
    -   `Series.where` and `Series.mask`: support for nullable arrays
        (e.g. pd.Int64Dtype)

-   Scikit-learn:

    -   Support for `sklearn.ensemble.RandomForestRegressor`
