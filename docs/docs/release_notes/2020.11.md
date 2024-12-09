Bodo 2020.11 Release (Date: 11/19/2020) {#November_2020}
========================================

This release includes many new features, bug fixes and performance
improvements. Overall, 126 code patches were merged since the last
release.

## New Features and Improvements

-   Bodo is updated to use Apache Arrow 2.0 (latest)

-   Performance and memory optimizations

    -   Significant memory usage optimizations for several operations
        involving string arrays
    -   Up to 2x speedup for many string operations such as
        `Series.str.replace/get/contains` and `groupby.sum()`

-   User-defined functions (UDFs)

    -   Support for returning datafarames from `DataFrame.apply()` and
        `Series.apply()`
    -   Support for returning nested arrays

-   Caching: for Bodo functions that receive CSV and JSON file names as
    string arguments, the cache will now be reused when file name
    arguments differ but have the same dataset type (schema).

-   Support for distributed deep learning with Tensorflow and PyTorch:
    <https://docs.bodo.ai/latest/source/dl.html>

-   Pandas coverage:

    -   Support for tuple values in Series and DataFrame columns
    -   Improvements to error checking and handling
    -   Automatic unrolling of loops over dataframe columns when
        necessary for type stability
    -   Support integer column names for Dataframes
    -   Support for `pd.Timedelta` values
    -   Support for `pd.tseries.offsets.DateOffset` and
        `pd.tseries.offsets.Monthend`
    -   Support for Series.dt, Timestamp, and DateTimeIndex attributes
        (`is_month_start`, `is_month_end`, `is_quarter_start`,
        `is_quarter_end`, `is_year_start`, `is_year_end`, `week`,
        `weekofyear`, `weekday`)
    -   Support for Series.dt and Timestamp `normalize` method
    -   Support for `Timestamp.components` and `Timestamp.strftime`
    -   Support for `Series.dt.ceil` and `Series.dt.round`
    -   Support for `pd.to_timedelta`
    -   Support `Series.replace` for *categorical* arrays where
        `value` and `to_replace` are scalars or lists
    -   Support for comparison operators on Decimal types
    -   Support for Series.add() with String, datetime, and timedelta
    -   Support for Series.mul() with string and int literal
    -   Support for setting values in *categorical* arrays
    -   Initial support for `pd.get_dummies()`
    -   Support for `Series.groupby()`

-   Scikit-learn: the following classes and functions are supported
    inside jit functions:

    -   `sklearn.linear_model.LinearRegression`
    -   `sklearn.linear_model.LogisticRegression`
    -   `sklearn.linear_model.Ridge`
    -   `sklearn.linear_model.Lasso`
    -   `sklearn.svm.LinearSVC`
    -   `sklearn.naive_bayes.MultinomialNB`
    -   `sklearn.metrics.accuracy_score`
    -   `sklearn.metrics.mean_squared_error`
    -   `sklearn.metrics.mean_absolute_error`

-   XGBoost: Training XGBoost model (with Scitkit-learn like API) is now
    supported inside jit functions:

    -   `xgboost.XGBClassifier`
    -   `xgboost.XGBRegressor`

    Visit <https://docs.bodo.ai/latest/source/ml.htmlfor more
    information about supported ML functions.

-   NumPy coverage:

    -   Support for `numpy.any` and `numpy.all` for all array types
    -   Support for `numpy.cbrt`
    -   Support for `numpy.linspace` arguments `endpoint`, `retstep`,
        and `dtype`
    -   `np.argmin` with axis=1
    -   Support for `np.float32(str)`

-   Support for `str.format`, `math.factorial`, `zlib.crc32`
