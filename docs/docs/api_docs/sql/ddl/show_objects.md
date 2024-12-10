# SHOW OBJECTS

Lists the tables and views for which you have access privileges for a specified database or schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-objects) for more details.

## Syntax

```sql
SHOW [ TERSE ] OBJECTS IN [<database_name>.]<schema_name>
```

## Usage notes

`SHOW OBJECTS` returns the following columns as `string` types unless otherwise mentioned:

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
- `OWNER_ROLE_TYPE`

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-objects) for descriptions of the columns.

For Iceberg catalogs, only a subset of the columns are supported. The rest will always be set to `NULL`. Supported columns are:

- `NAME`
- `SCHEMA_NAME`
- `KIND`
- `COMMENT`

The `TERSE` option will return only the following output columns, regardless of catalog:

- `CREATED_ON`
- `NAME`
- `SCHEMA_NAME`
- `KIND`

All columns will be of type `string`.