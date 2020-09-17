.. _September_2020:

Bodo 2020.09 Release (Date: 09/17/2020)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This release includes many new features, bug fixes and performance improvements.
Overall, 88 code patches were merged since the last release.


New Features and Improvements
-----------------------------

- Bodo is updated to use the latest versions of Numba, pandas and Arrow:

    - Numba 0.51.2
    - pandas 1.1.2
    - Arrow 1.0.1


- Major improvements in memory management. Bodo's memory consumption is reduced significantly
  by releasing memory as soon as possible in various operations such as Join, GroupBy, and Sort.

- Significant improvements in checking and handling various errors in I/O, providing clear error messages and graceful exits.

- Improvements in speed and scalability of `read_parquet` when reading from directories with large number of files.

- Distributed diagnostics is improved to provide clear messages on why a variable was assigned REP distribution.

- Improvements in caching support for I/O calls and groupby user-defined functions (UDFs).

- Support for more distributed getitem/setitem cases on arrays.

- Improvements on checking for unsupported functions and optional arguments.

- Significant performance improvements in groupby transformations (e.g. `GroupBy.cumsum`).

- Enhanced support for `DataFrame.select_dtypes`.

- Support for `axis=1` in `DataFrame.var/std`.

- Support for `Series.autocorr`.

- Support for `Series.is_monotonic_increasing/is_monotonic_decreasing`.

- Support `pd.Series()` constructor with a scalar data value.

- Support for `dayofweek`, `is_leap_year` and `days_in_month` in `Timestamp` and `Series.dt`.

- Support for `isocalendar` in `Series.dt` and `DatetimeIndex`.

- Support for `Series.cumsum/cummin/cummax`.

- Support for `Decimal` values in nested data structures.

- Improvements in table join performance.

- Support for `Series.drop_duplicates`.

- Support for `np.dot` and `@` operator on `Series`.

- Improvements in `pd.concat` support.

- Optimized `Series.astype(str)` for `int64` values.

- Support for `pd.Index` constructor.
