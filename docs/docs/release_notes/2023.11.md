
Bodo 2023.11 Release (Date: 11/07/2023) {#November_2023}
========================================

## New Features and Improvements

- BodoSQL generated plans have been further optimized to reduce compile time, runtime and memory usage, as well as
  resolve several bugs pertaining to stability.
- Support more functionality in BodoSQL:
  - `EQUAL_NULL` and `<=>` now supported on semi-structured data.
  -  Added support for the semi-structured functions `ARRAY_CONSTRUCT`, `ARRAYS_OVERLAP`, `ARRAY_POSITION`.
  - `CONCAT` (and `||`) now support concating binary data.
  - `LAST_DAY` now supports unquoted interval literals as arguments.
  - Support for writing semi-structured array columns and some cases of semi-structured object columns to Snowflake.
- Some casting functions that previously had incorrect behavior with nullable
  data are now fixed.

## 2023.11.1 New Features and Improvements


### Feature Updates:


  - Added the table function `FLATTEN` which can be used to explode a column of arrays alongside the `LATERAL` keyword. Currently allows ARRAY columns being passed in to the INPUT argument, as well as JSON columns under limited circumstances. Currently only allows the defaults for named arguments `PATH`, `OUTER`, `RECURSIVE` and `MODE`.
  - Added the function `OBJECT_CONSTRUCT_KEEP_NULL`, including the special syntax `OBJECT_CONSTRUCT_KEEP_NULL(*)`, as long as all of the key arguments are string literals.
  - Added the aggregation function `OBJECT_AGG` which takes in a column of strings and a column of data of any type and combines them into a JSON value where the first column is the keys and the second column is the values. Currently only supported within a GROUP BY clause.
  - Added the function `OBJECT_DELETE`. Non-literals key strings are partially supported.
  - Support `ARRAY_CONTAINS`, `ARRAY_UNIQUE_AGG`, `ARRAY_EXCEPT`, `ARRAY_INTERSECTION`, `ARRAY_CAT`, and `STRTOK_TO_ARRAY`.
  - Support for the table function `SPLIT_TO_TABLE`.
  - Support all sub-second interval units and their aliases.
  - Support all interval literal units without quotes.
  - Support unquoted date/time units in all functions that accept date/time units.
  - Support reading Object and deeply nested semi-structured columns from Snowflake.


### Improvements and Bug Fixes:


  - Enabled streaming in pipelines with semi-structured data.
  - Reduced out of memory risk during shuffle.
  - Optimized handling of nullable columns during SQL plan optimization.


### Dependency Updates:


  - Upgrade to Calcite 1.34
  - Upgrade to Pandas 2
  - Upgrade to Cython 3
  - Upgrade to HDF5 1.14

## 2023.11.2 New Features and Improvements


### Feature Updates:


  - General improvements to the spill-to-disk functionality.
  - Support `ARRAY_COMPACT`.
  - Support for indexing into array values via `GET()` and `arr[idx]`.
  - Added logging for failed metadata collection during planning. This is available with verbose_level >= 2.
  - Support for precision and scale arguments with to_number try_to_number, and aliases.
  - `DATE_FROM_PARTS`/`TIME_FROM_PARTS`/`TIMESTAMP_FROM_PARTS` & their aliases can take any numeric arguments, as opposed to only integers.


### Improvements and Bug Fixes:


  - Fixed bug where output of certain aggregate calls would have the wrong nullability.
  - Fixed various bugs that were limiting ability to call functions on columns of arrays of JSON data.
  - Fixed a bug in the Union (Distinct) operator which could lead to a hang in some cases.
  - Fixed `TO_ARRAY` and `ARRAY_TO_STRING` on column inputs and case statement.
  - Fixed shuffle corner case bug for semi-structured data array nulls.
  - Fixes to `COALESCE` typing behavior.


### Dependency Updates:


  - Bodo now uses PyArrow arrays to box/unbox nested arrays with zero-copy (made possible by Pandas 2 upgrade).
