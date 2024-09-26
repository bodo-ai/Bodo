Bodo 2020.10 Release (Date: 10/20/2020) {#October_2020}
========================================

This release includes many new features, bug fixes and performance
improvements. Overall, 117 code patches were merged since the last
release.

## New Features and Improvements

-   Initial support for Python classes using `bodo.jitclass` decorator.

-   

    Scikit-learn:

    :   -   

            Initial support for these scikit-learn classes:

            :   -   `sklearn.linear_model.SGDClassifier`
                -   `sklearn.linear_model.SGDRegressor`
                -   `sklearn.cluster.KMeans`

                For more information please refer to the documentation
                [here](https://docs.bodo.ai/latest/source/sklearn.html)

        -   Improved scaling of `RandomForestClassifier` training

-   Memory management and memory consumption improvements

-   

    Improvements for User-defined functions (UDFs):

    :   -   Compilation errors are now clearly shown for UDFs
        -   Support more complex UDFs (by running a full compiler
            pipeline)
        -   Support passing keyword arguments to UDF in
            `DataFrame.apply()` and `Series.apply()`
        -   Support much wider range of UDF types in `groupby.agg`

-   

    Connectors:

    :   -   Improved connector error handling
        -   Improved performance of `pd.read_csv` (further improvements
            in next release)
        -   `pd.read_parquet` supports column containing all NA (null)
            values

-   Caching: for Bodo functions that receive parquet file names as
    string arguments, the cache will now be reused when file name
    arguments differ but have the same parquet dataset type (schema).

-   Significantly improved the performance of merge/join operations in
    some cases

-   Support *for* loops over dataframe columns by automatic loop
    unrolling

-   Support using global dataframe/array values inside jit functions

-   Performance optimization for the `series.str.split().explode()`
    pattern

-   

    Pandas coverage:

    :   -   Support setting `df.columns` and `df.index`
        -   Support setting values in Categorical arrays
        -   `series.str.split`: added support for regular expression and
            `n` parameter
        -   `Series.replace` support for more array types
        -   Support `pd.series.dt.quarter`
        -   Support `series.str.slice_replace`
        -   Support `series.str.repeat`
        -   Improved support for `df.pivot_table` and `pd.crosstab`
        -   Support for `Series.notnull`
        -   Support integer label indexing for Dataframes and Series
            with RangeIndex
        -   Support setting `None` and `Optional` values for most arrays

-   

    NumPy coverage:

    :   -   Support for `np.union1d`
        -   `np.where`, `np.unique`, `np.sort`, `np.repeat`: support for
            Series and most array types
        -   Support `np.argmax` with `axis=1`
        -   Support for `np.min`, `np.max`, `min`, `max`, `np.sum`,
            `sum`, `np.prod` on nullable arrays
