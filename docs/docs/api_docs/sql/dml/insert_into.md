# INSERT INTO

Updates a table by inserting one or more rows into the table.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/insert) for more details.

Currently, BodoSQL only supports the following syntax:

```sql
INSERT INTO <name> [ ( <target_col_name> [ , ... ] ) ]
  {
    VALUES ( { <value> } [ , ... ] ) [ , ( ... ) ]  |
    <query>
  }
```
