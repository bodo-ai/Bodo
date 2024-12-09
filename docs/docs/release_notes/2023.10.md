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

### 2023.10.2 New Features and Improvements

- Fix critical bugs
- Reduce memory usage in GROUPBY

### 2023.10.3 New Features and Improvements

- Fix critical bugs.
- BodoSQL generated plans have been further optimized to reduce runtime.
- Support more functionality in BodoSQL:
  - Support `ARRAY_AGG` on strings and `ARRAY_AGG(DISTINCT)`
  - Support all call signatures for `TRUNC` and `CONCAT`
  - Support `current_database`
  - Support writing timezone aware data in output tables

### 2023.10.4 New Features and Improvements

- Fix critical runtime bugs.

### 2023.10.5 New Features and Improvements

- Fix critical runtime bugs.

### 2023.10.6 New Features and Improvements

- Fix critical runtime bugs.
- BodoSQL generated plans have been further optimized to reduce runtime and memory usage.
- Improved our ability to gather and use distinctness metadata 
- Support `DATEDIFF` between a timezone aware and a timezone naive column
- Support `DATE_PART`

### 2023.10.7 New Features and Improvements

- Fix critical runtime bugs.
- BodoSQL generated plans have been further optimized to reduce runtime and memory usage.
- Support more functionality in BodoSQL:
  - Support `JAROWINKLER_SIMILARITY`
  - Support `BASE64_ENCODE`
  - Support `BASE64_DECODE_STRING`
  - Support `BASE64_DECODE_BINARY`
  - Support `TRY_BASE64_DECODE_STRING`
  - Support `TRY_BASE64_DECODE_BINARY`
  - Support `HEX_ENCODE`
  - Support `HEX_DECODE_STRING`
  - Support `HEX_DECODE_BINARY`
  - Support `TRY_HEX_DECODE_STRING`
  - Support `TRY_HEX_DECODE_BINARY`
  - Support `ARRAY_SIZE`
  - Support `OBJECT_KEYS`
  - Support `getitem/isna` on NULL columns
  - Support `QUARTER` interval literals (and all aliases)
  - Support all call signatures for `TIMESTAMP_FROM_PARTS`
  - Support all call signatures for `TRY_TO_BOOLEAN/TRY_TO_DOUBLE`
  - Support all call signatures for `DATEADD/TIMEADD/TIMESTAMPADD`
