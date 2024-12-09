Bodo 2022.4 Release (Date: 4/29/2022) {#April_2022}
=====================================

This release includes many new features, usability and performance
improvements, and bug fixes. Overall, 60 code patches were merged since the last release.

## New Features and Improvements

-   Support for Python 3.10 (Conda/pip packages will be available soon)

-   Support for Pandas 1.4 (along with continued support for v1.3)

-   Connectors:

    -   When passing a list of paths to `pd.read_parquet`, the paths can be a combination
        of paths to files and glob strings.
    -   Improved performance of `pd.read_parquet` on remote filesystems when passing
        long lists of files.
    -   `DataFrame.to_parquet` now supports `row_group_size` parameter, which can be used to specify the maximum number
        of rows in generated row groups. Bodo now has a default row group size of 1M rows, to improve
        performance when reading the generated parquet datasets in parallel.
    -   `pd.read_parquet`: string columns can be forced to be read with dictionary encoding
        by passing a list of column names with `_bodo_read_as_dict` parameter.
    -   Support for S3 anonymous access with `storage_options={"anon": True}`
        in `pd.read_csv` and `pd.read_json`
    -   Improved performance and memory utilization of `pd.read_csv` at compilation and
        run time (especially when reading first n rows from remote filesystems such as S3)

-   Parallel support for `pd.date_range`: Bodo automatically creates a date range
    that is distributed across processes

-   Improved performance of `Series.str.startswith/endswith/contains` for dictionary-encoded string arrays

-   Reduced compilation time for `DataFrame.memory_usage()`

-   Reduced compilation time when using `pandas.read_sql()` with wide tables.

-   Pandas:

    -   Support for `DataFrame.melt()` and `pd.melt()`
