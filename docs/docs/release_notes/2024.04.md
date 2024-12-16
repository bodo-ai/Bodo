Bodo 2024.4 Release (Date: 4/11/2024) {#April_2024}
=====================================

### New Features:

- Added support for function `#!sql CURRENT_ACCOUNT`, which requires a Snowflake Catalog.
- Executing Snowflake Javascript UDFs in Bodo is now supported for all types except TIMESTAMP and Semi-Structured types.
- Added support for automatically generating lower_bound, upper_bound, value_counts, and null_value_counts for Iceberg data files written by Bodo.
- Improved support for handling skewed joins to avoid out of memory errors.
- Experimental support for TIMESTAMP_TZ

### Performance Improvements:

- Adds support for generating runtime filters on the probe side of joins based upon the values encountered on the build side. In many cases this will significantly improve performance.

### Bug Fixes:

- Added support for additional types in Iceberg Filter pushdown
- Fixed a corner case when reading non-nullable VARIANT data from Snowflake

### Dependency Upgrades:
- Upgraded Numba to 0.59.1.


## 2024.4.1 New Features and Improvements


### New Features:
- Support reading and writing Iceberg tables on Azure.
- Support for DROP TABLE on Snowflake and Iceberg.
- Minor update to add more filter pushdown information for logging level >= 2

### Performance Improvements:
- Support pushing join runtime filters past the partition by columns for window functions.

### Bug Fixes:
- Fixed an issue using EC2 instance profile authentication in BodoSQL FileSystemCatalogs
- Fixed the expected column names for INSERT INTO queries
- Handling for extra whitespace in string inputs for TO_BOOLEAN, TO_NUMBER, and TO_DOUBLE
- Fixed a compilation error when join runtime filters occur after window functions with verbose mode enabled
- Fixed an issue in filter pushdown that could result in an runtime error


