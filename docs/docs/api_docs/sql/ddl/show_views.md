# SHOW VIEWS

Lists the views for which you have access privileges for a specified database or schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-views) for more details.

## Syntax

```sql
SHOW [ TERSE ] VIEWS IN [<database_name>.]<schema_name>
```

## Usage notes

`SHOW VIEWS` returns the following columns as `string` types unless otherwise mentioned:

- `CREATED_ON`
- `NAME`
- `RESERVED`
- `SCHEMA_NAME`
- `COMMENT`
- `OWNER`
- `TEXT`
- `IS_SECURE`
- `IS_MATERIALIZED`
- `OWNER_ROLE_TYPE`
- `CHANGE_TRACKING`

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-tables) for descriptions of the columns.

For Iceberg catalogs, only a subset of the columns are supported. The rest will always be set to `NULL`. Supported columns are:

- `NAME`
- `SCHEMA_NAME`
- `COMMENT`

The `TERSE` option will return only the following output columns, regardless of catalog:

- `CREATED_ON`
- `NAME`
- `SCHEMA_NAME`
- `KIND`

All columns will be of type `string`.