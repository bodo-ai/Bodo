Bodo 2021.1 Release (Date: 1/26/2021) {#January_2021}
=====================================

This release includes many new features, bug fixes and performance
improvements. Overall, 61 code patches were merged since the last
release.

## New Features and Improvements

-   Connectors:

    -   Support *filter pushdown* when reading partitioned parquet
        datasets: at compile time, Bodo detects if filters are applied
        to a dataframe after `read_parquet`, and generates code that
        applies those filters at read time so that only the required
        parquet files are read.
    -   Support for `Series.to_csv()`
    -   Supports passing `file` and `dtype` arguments of `np.fromfile`
        as kwargs.

-   Support for f-strings in Bodo jitted functions

-   Support passing Bodo distributed JIT functions to other Bodo JIT
    functions

-   Pandas coverage:

    -   Support groupby with `pd.NamedAgg()`
    -   Support for `groupby.size`
    -   Support for `groupby.shift`
    -   Match input row order of pandas in `groupby.apply` when
        applicable
    -   Support `min_periods` in rolling calls
    -   Support passing a dictionary of data types to `df.astype()`
    -   Support dataframe setitem of multiple columns. For example:
        `df[["A", "B"]] = 1.3`
    -   Support for `Index.get_loc()`
    -   Support `ddof` argument (delta degrees of freedom) of
        `Series.cov`
    -   Support `Series.is_monotonic` property
    -   Initial support for dictionaries in `Series.replace`
    -   Support `Series.reset_index(drop=True)`
    -   Support level argument with all levels in `reset_index()`
    -   Several documentation improvements

-   Scikit-learn:

    -   Support for `sklearn.model_selection.train_test_split` inside
        jit functions.
    -   Support for `sklearn.preprocessing.MinMaxScaler` inside jit
        functions.
