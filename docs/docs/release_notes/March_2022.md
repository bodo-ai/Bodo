Bodo 2022.3 Release (Date: 3/31/2022) {#March_2022}
=====================================

This release includes many new features, usability and performance
improvements, and bug fixes. Overall, 74 code patches were merged since the last release.

## New Features and Improvements

-   Bodo is updated to use Arrow 7.0 (latest)

-   Initial support for dictionary-encoded string arrays.
    Dictionary encoding can improve performance and reduce memory usage significantly
    when data has many repeated values which is common in practice (see [here](https://arrow.apache.org/docs/format/Columnar.html#dictionary-encoded-layout){target=blank}).
    Bodo now uses dictionary encoding automatically in `pd.read_parquet` when a string column can benefit from it.
    Join, sort and parquet write operations support dictionary-encoded string arrays as well, and
    the support will expand to others in the future.
    Bodo will fall back to regular string arrays automatically if an
    operation does not support dictionary encoding.

-   Connectors:

    -   `pd.read_parquet` performance improvements when multiple processes read
        from the same file.
    -   Support for filter pushdown in Parquet and Snowflake when using `Series.isin`
    -   Support for SparkSQL's `input_file_name` functionality for `read_parquet` using a new `_bodo_input_file_name_col` argument.
    -   Support for `chunksize` in `pd.to_sql`
    -   Optimized `df.to_parquet` memory usage when writing string columns
    -   Support for passing list of columns as `columns` parameter of `df.to_csv`
    -   Support in `pd.read_sql` for returning an empty DataFrame from Snowflake, either
        due to an empty query or the result of filter pushdown.
    -   Changed default value of `orient` and `lines` in `DataFrame.to_json` to `records` and `True` respectively to enable parallel write (Pandas uses `columns` and `False` as default).

-   Bodo now provides compiler optimization logging through `bodo.set_verbose_level()`.
    This can be used to display certain optimizations performed at compile time,
    such as filter pushdown, column pruning, and which columns are read with dictionary
    encoding when reading from Parquet. See [Verbose Mode][bodoverbosemode] for more details. 

-   Improvements in error checking and quality of error messages.

-   Avoid hang when encountering unhandled exceptions on a single process.

-   Introduced `replicated` JIT decorator flag (opposite of `distributed`).
-   If the user provided `distributed` JIT flag for some input and return values but not all, bodo can now infer distribution of the rest.

-   Performance optimizations:

    -   Improved memory usage during parallel `groupby.apply`
    -   Improved `df.sample` performance when `frac=1` and `replace=False`

-   Pandas:

    -   Initial support for Timezone-Aware arrays and timestamps
        -   Added support for `array.tz_convert`, `Series.dt.tz_convert`, `Timestamp.tz_convert`, `DatetimeIndex.tz_convert`, `Timestamp.tz_localize`
    -   Support for `Series.str.cat`
    -   Support for `pd.unique` on Series and 1-D arrays
    -   Support for comparison operators between `DatetimeIndex` and `pd.Timestamp`
        and `TimedeltaIndex` and `pd.Timedelta`
    -   Support for `DataFrame.set_index` on single-column DataFrames
    -   Support for `Series.first_valid_index` and `Series.last_valid_index`
    -   Support for conversion between `pd.timestamp` and `np.datetime64`
