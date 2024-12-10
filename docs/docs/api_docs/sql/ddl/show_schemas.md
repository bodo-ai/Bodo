# SHOW SCHEMAS

Lists the schemas for which you have access privileges for a specified database.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-schemas) for more details.

## Syntax

```sql
SHOW [ TERSE ] SCHEMAS IN <database_name>
```

## Usage notes

`SHOW SCHEMAS` returns the following columns as `string` types unless otherwise mentioned:

- `CREATED_ON`
- `NAME`
- `IS_DEFAULT`
- `IS_CURRENT`
- `DATABASE_NAME`
- `OWNER`
- `COMMENT`
- `OPTIONS`
- `RETENTION_TIME`
- `OWNER_ROLE_TYPE`

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-schemas) for descriptions of the columns.

For Iceberg catalogs, only a subset of the columns are supported. The rest will always be set to `NULL`. Supported columns are:

- `NAME`
- `DATABASE_NAME`

The `TERSE` option will return only the following output columns, regardless of catalog:

- `CREATED_ON`
- `NAME`
- `SCHEMA_NAME`
- `KIND`

All columns will be of type `string`.
