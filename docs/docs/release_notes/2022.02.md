Bodo 2022.2 Release (Date: 2/28/2022) {#February_2022}
=====================================

This release includes many new features and usability improvements.
Overall, 82 code patches were merged since the last release.

## New Features and Improvements

-   Reduced the import time of the Bodo package substantially

-   Bodo is now available with `pip` on x86 Mac

-   Bodo is upgraded to use Numba 0.55.1 (the latest release)

-   Bodo is upgraded to use scikit-learn v1

-   Bodo now supports MPICH version 3.4

-   Connectors:

    -   `pd.read_sql`: Support and getting start documentation for for Oracle DB
        and PostgreSQL.
    -   `pd.read_parquet` now supports glob patterns
    -   Support for `escapechar` argument in `pd.read_csv`
    -   Decreased compilation time when reading wide schemas with 1000s
        of columns usings `pd.read_parquet`.
    -   Optimized runtime of `pd.read_parquet` with `head(0)` to skip
        any unnecessary schema collection for each parquet file and just
        look at the metadata. This optimization is helpful when loading
        a DataFrame schema.
    -   Support using filter pushdown with a single filter consisting of
        `Series.isna`, `Series.isnull`, `Series.notna`, or `Series.notnull`.
    -   Full filter pushdown support with `hdfs` and `gcs` using `pd.read_parquet`
    -   Improved performance and error handling when using `DataFrame.to_sql`
        with Snowflake.
    -   Bodo now prints a warning if the number of Parquet row groups is too small for effective parallel I/O.


-   Support for using lists and sets as constant global values.

-   Support for distributed global dataframe values

-   Added a compiler optimizations for forcing the columns in a DataFrame
    to match a DataFrame with an existing schema via `DataFrame.dtypes`.
    In particular when Bodo encounters code like:

    ``` ipython3
    @bodo.jit
    def f(df1, df2):
        return df1.astype(df2.dtypes)
    ```

    Bodo will automatically use the internal Bodo types for all columns in
    `df2`. This enables using astypes for conversions that are typically not
    possible in Pandas because the column has an `object` dtype. For example,
    this can be used to convert a column from `datetime64[ns]` to
    `datetime.date` with `astype`.

-   Improved runtime performance when copying a string data from one array to
    another or when computing an array of string lengths.


-   Pandas:


    -   Support for passing multiple columns to `values` and `index` with
        `DataFrame.pivot()` and `DataFrame.pivot_table()`
    -   Support for using `pd.pivot()` and `pd.pivot_table()`. Functionality
        is equivalent to `DataFrame.pivot()` and `DataFrame.pivot_table()`
    -   Support for `DataFrame.explode()`
    -   Support for `DataFrame.where()` and `DataFrame.mask()`
    -   Support for `Series.duplicated()` and `Index.duplicated()`.
    -   Support for `Series.rename_axis()`
    -   Support for using `object` in `DataFrame.astype`. Bodo doesn't
        have a generic "object" type, so the type of the column remains
        the same.
