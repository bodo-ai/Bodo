# ALTER TABLE

Modifies the properties, columns, or constraints for an existing table from the current/specified schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/alter-table) for more details.

Currently, BodoSQL only supports the following syntax:

```sql
ALTER TABLE [ IF EXISTS ] <name> RENAME TO <new_table_name>
```
