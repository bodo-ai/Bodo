Bodo 2024.3 Release (Date: 03/12/2024) {#March_2024}
========================================

### New Features:

- Added parquet row count collection with TablePath API in BodoSQL to improve generated plans
- Introduced a FileSystemCatalog to enable treating a local file system or s3 bucket as a database.
- BodoSQL now supports streaming Parquet write with the FileSystemCatalog.
- Initial support for reading Iceberg tables that have gone through schema evolution. Full support will be added in the next release.
- Enable decorrelating subqueries that reference columns from multiple tables.


### Performance Improvements:

- Added an optimization to reduce memory usage and runtime of min row number filter aggregations.
- Improved simplification of certain expressions in the planner.

### Bug Fixes:
- Reduced likelihood of Bodo exceeding timeouts while attempting to probe Snowflake for the true type of a semi-structured column on extremely large tables.
- Bodo now correctly writes Iceberg Field IDs as Parquet Field IDs in the generated Parquet files for Iceberg tables.
- Resolved a stack overflow issue with extremely complex SQL plans


### Dependency Upgrades:
- Upgraded to Python 3.12
- Upgraded to Numba 0.59
- Upgraded to Arrow 15
- Upgraded to Pandas 2.1


## Bodo 2024.3.1 Release

### New Features:

- Support the newer `scipy.fft` API over `scipy.fftpack`
- Full support for reading Iceberg tables that have gone through schema evolution.

### Performance Improvements:

- Enabled better filter optimizations for Left and Right joins.
- Improved planner optimization on most `to_<type>` conversion functions to become equivalent to casts.