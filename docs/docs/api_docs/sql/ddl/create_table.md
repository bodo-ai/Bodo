# CREATE TABLE

Creates a new table in the current/specified schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/create-table) for more details.

## Syntax

Currently, BodoSQL only supports the `CREATE TABLE ... AS SELECT` (CTAS) form, and only with the following syntax:

```sql
CREATE [ OR REPLACE ] TABLE [ IF NOT EXISTS ] <name>
[ ( <col_name> [ <col_type> ] , <col_name> [ <col_type> ] , ... ) ]
[ COMMENT = <string> ]
AS <query>
```

BodoSQL can parse additional forms of `#!sql CREATE TABLE` syntax, though they currently do not have any effects:

- The `#!sql TRANSIENT` keyword
- The `#!sql LOCAL`, `#!sql GLOBAL`, `#!sql TEMP`, `#!sql TEMPORARY` or `#!sql VOLATILE` keywords
- A `#!sql COPY GRANTS` clause
- A `#!sql CLUSTER BY` clause
