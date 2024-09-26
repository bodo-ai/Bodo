Bodo 2020.08 Release (Date: 08/21/2020) {#August_2020}
========================================

This release includes many new features, bug fixes and performance
improvements. Overall, 112 code patches were merged since the last
release.

## New Features and Improvements

-   Bodo is updated to use the latest versions of Numba, pandas and
    Arrow:

    -   Numba 0.51.0
    -   pandas 1.1.0
    -   Arrow 1.0

-   Support reading and writing Parquet files with columns where values
    are arrays or structs, which can contain other arrays/structs with
    arbitrary nesting.

-   S3 I/O: automatically determine the region of the S3 bucket when
    reading and writing.

-   Initial support for scikit-learn RandomForestClassifier (fit,
    predict and score methods)

-   Support `sklearn.metrics.precision_score`,
    `sklearn.metrics.recall_score` and `sklearn.metrics.f1_score`.

-   Improved caching support (caching `@bodo.jit` functions with
    cache=True)

-   Initial support for arrays of map data structures

-   Support `count` and `offset` arguments of `np.fromfile`

-   New `bodo.rebalance()` function for load balancing dataframes
    manually if desired

-   Support setting dataframe column as attribute, for example:
    `df.B = "AA"`

-   Support DataFrame min/max/sum/prod/mean/median functions with
    `axis=1`

-   Support `df.loc[:,columns]` indexing

-   `pd.concat` support for mix of Numpy and nullable integer/bool
    arrays

-   Support parallel append to dataframes (concatenation reduction)

-   Support `GroupBy.idxmin` and `GroupBy.idxmax`

-   Improvements and optimizations in user-defined function (UDF)
    handling

-   Basic support for `Series.where()`

-   Support calling bodo.jit functions inside prange loops

-   Support `DataFrame.select_dtypes` with constant strings

-   Support `DataFrame.sample`

-   Support `Series.replace()` and `df.replace()` (scalars and lists)

-   Support for Series.dt methods: `total_seconds()` and
    `to_pytimedelta()`

-   Improved support for Categorical data types

-   Support for `pandas.Timestamp.isocalendar()`

-   Support `np.digitize()`

-   Improved error handling during I/O when input CSV or Parquet file
    does not exist

-   Support pd.concat(axis=1) for dataframes

-   Significant improvements in compilation time for dataframes with
    large number of columns

-   `bodo.is_jit_execution()` can be used to know if a function is
    running with Bodo.
