Bodo 2021.10 Release (Date: 10/28/2021) {#October_2021}
========================================

This release includes many new features, optimizations, bug fixes and
usability improvements. Overall, 71 code patches were merged since the
last release.

## New Features and Improvements

-   The Bodo Community Edition can now run on up to 8 cores.

-   Bodo is updated to use Numba 0.54.1 (latest).

-   Improved error messages and documentation.

-   Connectors:

    -   `pandas.read_csv`: support for `chunksize` and `nrows`
        parameters
    >
    -   Snowflake:
    >
        -   Improved performance and scalability using the new
            parallel fetch functionality of Snowflake's Python
            connector, which retrieves data as batches of Arrow
            tables
        -   Support for removing unused columns from the SQL query.
        -   Support for filter pushdown of Pandas comparison
            operations into the SQL query.

-   Reduced compilation time for `DataFrame.describe`

-   Pandas:

    -   `DataFrame.sort_values`: supports passing `na_position` as a
        list with one value per column.

BodoSQL 2021.10beta Release (Date: 10/28/2021)

This release includes SQL bug fixes, increased SQL coverage and various
usability improvements. Overall, 27 code patches were merged since the
last release.

## New Features and Improvements

-   Improved error messages with expanded documentation.

-   Support for passing `CategoricalArray` and
    `DateArray` to BodoSQL. BodoSQL will automatically
    convert these arrays to supported types.

-   SQL Coverage

    This release added the following additional SQL coverage to BodoSQL.
    Please refer to our
    [documentation](https://docs.bodo.ai/latest/source/programming_with_bodo/BodoSQL.html#supported-operations)
    for more details regarding usage.

    -   Support for `TO_DATE` function
    -   Support for string column casting inside
        `DATE_ADD` and `DATE_SUB`
    -   Support for `nulls first` and `nulls last` inside `order by`.
    -   Support for String columns in Window Aggregation Functions.
    -   Provided more efficient implementations for `NVL` and `IFNULL`
        when there is a column and a scalar.
