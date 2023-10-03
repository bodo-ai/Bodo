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
