# CREATE VIEW

Creates a new view in the current/specified schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/create-view) for more details.

## Syntax

Currently, BodoSQL only supports the following syntax:

```sql
CREATE [ OR REPLACE ] VIEW [ IF NOT EXISTS ] <name> 
[ ( <column_list> ) ]
as <select statement>
```

BodoSQL can parse additional forms of `#!sql CREATE VIEW` syntax purely for the purposes of allowing view inlining:

- The `#!sql SECURE` keyword
- The `#!sql LOCAL`, `#!sql GLOBAL`, `#!sql TEMP`, `#!sql TEMPORARY` or `#!sql VOLATILE` keywords
- The `#!sql RECURSIVE` keyword
- A `#!sql COPY GRANTS` clause
- A view-level `#!sql COMMENT`
- A view-level `#!sql WITH TAG` clause
- A view-level `#!sql ROW ACCESS POLICY` clause
