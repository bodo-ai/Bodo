# SHOW VIEWS

Lists the views for which you have access privileges for a specified database or schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/show-views) for more details.

Currently, BodoSQL only supports the following syntax:

```sql
SHOW TERSE VIEWS IN [<database_name>.]<schema_name>
```