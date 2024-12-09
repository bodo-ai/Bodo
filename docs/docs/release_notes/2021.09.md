Bodo 2021.9 Release (Date: 9/29/2021) {#September_2021}
========================================

This release includes many new features, optimizations, bug fixes and
usability improvements. Overall, 98 code patches were merged since the
last release.

## New Features and Improvements

-   Bodo is updated to use Numba 0.54 (latest)

-   Performance improvements:

    -   Significantly improved the performance and scalability of
        parallel `merge` and `join` operations
    -   Improved the performance and scalability of `groupby.nunique`
    -   General performance improvements for operations involving data
        shuffling
    -   Optimized many compilation paths, especially those involving
        DataFrames. This will lead to shorter compilation times for
        many use cases.
    -   Optimizations in `pd.read_sql` to limit the data read when
        `LIMIT` is provided.

-   Pandas:

    -   Support for `Series.shift` on timedelta64 data
    -   Support for `pd.cut()` and `pd.qcut()`
    -   Support for `first`, `last`, `median`, `nunique`, `prod`, and
        `var` in `groupby.transform`
    -   Support for multiplication with DateOffset
    -   Support for `Series.round()` on nullable integers
    -   Support for `to_strip` argument in
        `series.str.strip/lstrip/rstrip`
    -   Increased Binary Array/Series/DataFrame support. In particular:
        -   Support for `first`, `last`, `shift`, `count`, `nunique`,
            `size`, `value_counts` for Binary Series and DataFrames.
        -   Groupby support with binary keys/values.
        -   Support for `sort_values` with binary columns.
        -   Join with binary keys
        -   Most generic Series/DataFrame operations.
    -   Support for equi-join with additional non-equi-join conditions
        through our general merge condition syntax. Please refer to the
        documentation for more information.

BodoSQL 2021.9beta Release (Date: 9/29/2021)

This release adds SQL bug fixes and various usability improvements,
including a reduced package size. BodoSQL users should also benefit from
compilation time improvements due to improvements in the engine.
Overall, 25 code patches were merged since the last release.

## New Features and Improvements

-   Decreased package size and removed external dependencies.

-   Improved error messages with shortened stack traces.

-   SQL Coverage

    This release added the following additional SQL coverage to BodoSQL.
    Please refer to our documentation for more details regarding usage.

    -   Support for `UTC_TIMESTAMP` function
    >
    -   Support for `UTC_DATE` function
    >
    -   Support for `PIVOT`
    >
    -   Support for the following Window Functions:
    >
        -   `MAX`
        -   `MIN`
        -   `COUNT/COUNT(*)`
        -   `SUM`
        -   `AVG`
        -   `STDDEV`
        -   `STDDEV_POP`
        -   `VARIANCE`
        -   `VAR_POP`
        -   `LEAD`
        -   `LAG`
        -   `FIRST_VALUE`
        -   `LAST_VALUE`
        -   `NTH_VALUE`
        -   `NTILE`
        -   `ROW_NUMBER`
