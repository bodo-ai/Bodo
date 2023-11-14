
Bodo 2023.11 Release (Date: 11/07/2023) {#November_2023}
========================================

## New Features and Improvements

- BodoSQL generated plans have been further optimized to reduce compile time, runtime and memory usage, as well as
  resolve several bugs pertaining to stability.
- Support more functionality in BodoSQL:

  - `EQUAL_NULL` and `<=>` now supported on semi-structured data
  -  Added support for the semi-structured functions `ARRAY_CONSTRUCT`, `ARRAYS_OVERLAP`, `ARRAY_POSITION`
  - `CONCAT` (and `||`) now support concating binary data
  - `LAST_DAY` now supports unquoted interval literals as arguments
  - Support for writing semi-structured array columns and some cases of semi-structured object columns to Snowflake
- Some casting functions that previously had incorrect behavior with nullable
  data are now fixed.
