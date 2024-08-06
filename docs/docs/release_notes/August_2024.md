Bodo 2024.8 Release (Date: 8/6/2024) {#August_2024}
=====================================

### New Features:

* Expanded decimal support:
    * Added native decimal support for addition/subtraction.
    * Added native decimal support for `MEDIAN` with `GROUP BY`.
    * Added native decimal support for the following numeric functions: `ATAN`, `ATAN2`, `ATANH`, `COS`, `COSH`, `COT`, `DEGREES`, `RADIANS`, `SIN`, `SINH`, `TAN`, `TANH`, `EXP`, `POWER`, `LOG`, `LN`, `SQRT`, `SQUARE`, `ROUND`, `ABS`, `DIV0`, `SIGN`.
    * Support for binary operations between decimal and integer.
* Extended use of low-ndv IN join filters to Iceberg.
* Generate Join Filters with empty build tables and Snowflake IO.
* Support for Join Filters with Interval Joins.
* Streaming support for several more aggregate patterns, especially with `PIVOT`.
* Streaming support for grouping sets without the empty set.
* Support Create Table As Select (CTAS) with `TABLE COMMENTS`, `COLUMN COMMENTS`, and `TBLPROPERTIES` for Iceberg.

### Performance Improvements:

* Revamp shuffle in SQL join and groupby to use non-blocking MPI which can improve performance significantly.
* Improved streaming performance for more rank window functions (`RANK`, `PERCENT_RANK`, `CUME_DIST`, `ROW_NUMBER`).
* Added streaming support for `MAX(X) OVER ()`, `MIN(X) OVER ()`, `SUM(X) OVER ()`, and `COUNT(X) OVER ()`.
* Expanded sort for the streaming sort based implementation of window functions to all array types.
* Started decomposing `AVG`, var/std functions, covar functions, and `CORR` into combinations of sum/count to both allow for decimal support, and also allow for increased re-use of computations when multiple such functions are present.
* Significant improvements to Iceberg Scan Planning performance such as fetching the parquet metadata in parallel.
* Improved shuffle performance for variable length types (string/array).
* Improved BodoSQL’s ability to prune empty plan sections.
* Improved performance of decimal to string and decimal to double casting.

### Bug Fixes:

* Fixed bug related to string columns in streaming join sometimes causing segmentation faults.
* Fixed bug when calling `DATEADD` (and similar functions) with month/quarter/year units when the input date was a leapday.
* Fixed bug in verbose mode that would cause incorrect timer information to be displayed for non-streaming rel nodes.
* Fixed bug writing where writing to an iceberg table with a non-iceberg type (`int8`/`uint8`) caused an error instead of an upcast.
* Fixed a bug that would cause `SHOW TABLES` commands to error when the tables had a large number of rows/bytes.
* Fixed a bug in comparisons for elements of nested arrays with dictionary-encoded data.

### Dependency upgrade:

* Upgraded to Numba 0.60.
