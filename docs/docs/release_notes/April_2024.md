Bodo 2024.4 Release (Date: 4/11/2024) {#April_2024}
=====================================

## New Features:

- Added support for function `#!sql CURRENT_ACCOUNT`, which requires a Snowflake Catalog.
- Executing Snowflake Javascript UDFs in Bodo is now supported for all types except TIMESTAMP and Semi-Structured types.
- Added support for automatically generating lower_bound, upper_bound, value_counts, and null_value_counts for Iceberg data files written by Bodo.
- Improved support for handling skewed joins to avoid out of memory errors.
- Experimental support for TIMESTAMP_TZ

## Performance Improvements:

- Adds support for generating runtime filters on the probe side of joins based upon the values encountered on the build side. In many cases this will significantly improve performance.

## Bug Fixes:

- Added support for additional types in Iceberg Filter pushdown
- Fixed a corner case when reading non-nullable VARIANT data from Snowflake

## Dependency Upgrades:
- Upgraded Numba to 0.59.1.
