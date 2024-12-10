# DROP TABLE

Removes a table from the current/specified schema.

BodoSQL supports the optional keyword `PURGE` which specifies that the underlying data and metadata files should be deleted, as opposed to just making the table no longer accessible.

The effect of this keyword depends on which catalog is used.
For a Snowflake catalog, it is a no-op since Snowflake does not currently support the purge command.
For Iceberg catalogs, the command may tell the catalog that the data files are marked for deletion, but they may or may not be deleted right away.

The optional keyword `CASCADE` or `RESTRICT` is a no-op in Iceberg catalogs, but it is used in Snowflake to specify whether the table can be dropped if foreign keys exist that reference the table.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/drop-table) for more details.

## Syntax

Currently, BodoSQL only supports the following syntax:

```sql
DROP TABLE [ IF EXISTS ] <name> [ CASCADE | RESTRICT ] [ PURGE ]
```
