# SHOW TABLES

Lists the tables for which you have access privileges for a specified database or schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-tables) for more details.

## Syntax

```sql
SHOW [ TERSE ] TABLES IN [<database_name>.]<schema_name>
```

## Usage notes

`SHOW TABLES` returns the following columns as `string` types unless otherwise mentioned:

- `CREATED_ON`
- `NAME`
- `SCHEMA_NAME`
- `KIND`
- `COMMENT`
- `CLUSTER_BY`
- `ROWS` - _type_ `Decimal(38, 0)`
- `BYTES` - _type_ `Decimal(38, 0)`
- `OWNER`
- `RETENTION_TIME`
- `AUTOMATIC_CLUSTERING`
- `CHANGE_TRACKING`
- `IS_EXTERNAL`
- `ENABLE_SCHEMA_EVOLUTION`
- `OWNER_ROLE_TYPE`
- `IS_EVENT`
- `IS_HYBRID`
- `IS_ICEBERG`
- `IS_IMMUTABLE`

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-tables) for descriptions of the columns.

For Iceberg catalogs, only a subset of the columns are supported. The rest will always be set to `NULL`. Supported columns are:

- `NAME`
- `SCHEMA_NAME`
- `KIND`
- `COMMENT`
- `IS_ICEBERG`

The `TERSE` option will return only the following output columns, regardless of catalog:

- `CREATED_ON`
- `NAME`
- `SCHEMA_NAME`
- `KIND`

All columns will be of type `string`.