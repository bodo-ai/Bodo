# ALTER VIEW

Modifies the properties for an existing view from the current/specified schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/alter-view) for more details.

## Syntax

Currently, BodoSQL only supports the following syntax:

```sql
ALTER VIEW [ IF EXISTS ] <name> RENAME TO <new_name>
```
