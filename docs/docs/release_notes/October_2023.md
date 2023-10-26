Bodo 2023.10 Release (Date: 10/02/2023) {#October_2023}
========================================

## New Features and Improvements

- BodoSQL generated plans have been further optimized to reduce runtime and memory usage, as well as
  resolve several bugs that would prevent a plan from being produced. These changes include but are
  not limited to the following:
  - Improved ability to push down filters
  - Improved quality of join orderings based on meta data
  - Improved simplification of expressions and constant folding to reduce computations
- Unreserve several keywords in the BodoSQL parser, allowing them to be column names, aliases,
  or table names: `ROW_NUMBER`, `INTERVAL`, `PERCENT`, `COUNT`, `TRANSLATE`, `ROLLUP`, `MATCHES`, 
  `ABS`, `LAG`, and `MATCH_NUMBER`.
- Allow a BodoSQL Snowflake Catalog to be created from a connection string using `bodosql.SnowflakeCatalog.from_conn_str`.
- Support `ANY_VALUE` on array data

### 2023.10.1 New Features and Improvements

- Fix critical runtime bugs in vectorized execution mode.
- BodoSQL generated plans have been further optimized to reduce runtime and memory usage.
- Better compile time evaluation for datetime operations inside the planner.
- Major version upgrades
  - Upgrade Python to 3.11
  - Upgrade Numba to 0.57
  - Upgrade Calcite to 1.31
  - Upgrade Iceberg to 1.3.1
  - Upgrade Pandas to 1.5