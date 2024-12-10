Bodo 2022.9 Release (Date: 09/31/2022) {#September_2022}
========================================

## New Features and Improvements

Compilation / Performance improvements:

- Passing string data from Bodo JIT to Python and back (boxing/unboxing) is now much faster using the new Arrow support in Pandas. Dictionary-encoded (compressed) string arrays stay dictionary-encoded between calls.
- Optimized `pd.to_numeric()` for compressed string data.
- Support for compressed strings in `read_csv()` using user-specified argument (`“_bodo_read_as_dict“`).


I/O:

- Support for loading no data columns from Iceberg and Snowflake when just returning the length of a table.
- Support for limit pushdown with Snowflake.
- Update the verbose logging API to track limit pushdown with verbose level 1.


Iceberg:

- Support for appending to Iceberg tables with pre-defined partition spec and/or sort-order.
- Support for compressed string read from Iceberg tables.


BodoSQL:

- Introduced the `SnowflakeCatalog` object so users can connect their Snowflake account to BodoSQL easily. When added to a `BodoSQLContext`, BodoSQL will directly search and load tables from inside Snowflake. For more information please refer to the documentation.
- Added `BodoSQLContext` methods `add_or_replace_view`, `remove_view`, `add_catalog`, and `remove_catalog` for creating an updated `BodoSQLContext`.
- BodoSQL now pushes limits in front of projections/element-wise functions to enable limit pushdown in most queries.
- If passing unsupported types to BodoSQL, BodoSQL will now attempt to process the query without using those columns. This enables compilation when using only the columns in the table with supported types.
- `LEAD` and `LAG` now support an optional fill value argument, and the explicit `RESPECT_NULLS` syntax.
  - Support for explicitly passing `NULL` for the fill value.
- Support for issuing a delete query in Snowflake using `SnowflakeCatalog`. This works by pushing the entire query directly into Snowflake.
- Support for `ILIKE` operator
- Support for `CONTAINS` operator
- Support for `MEDIAN` aggregate function
- Support for `SQUARE`, `CBRT`, `FACTORIAL` functions
- Support for aliases `VARIANCE_POP` and `VARIANCE_SAMP`.
- Support for `NEXT_DAY` and `PREVIOUS_DAY`.
