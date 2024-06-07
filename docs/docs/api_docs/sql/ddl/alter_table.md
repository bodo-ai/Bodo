# ALTER TABLE

Modifies the properties, columns, or constraints for an existing table from the current/specified schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/alter-table) for more details.

Currently, BodoSQL only supports the following operations:

## Renaming a table
```sql
ALTER TABLE [ IF EXISTS ] <name> RENAME TO <new_table_name>
```

## Setting / unsetting table properties


`ALTER TABLE SET` is used to set table-wide properties. If a particular property was already set, this overrides the old value with the new one.
`ALTER TABLE UNSET` is used to drop table properties.

!!! note
    This operation is currently only supported for Iceberg.


```sql
ALTER TABLE [ IF EXISTS ] <name> 
    SET ( PROPERTY | PROPERTIES | TAG | TAGS | TBLPROPERTY | TBLPROPERTIES ) 
    '<tag_name>' = '<tag_value>' [ , '<tag_name>' = '<tag_value>' ... ]
```

```sql
ALTER TABLE [ IF EXISTS ] <name> 
    UNSET ( PROPERTY | PROPERTIES | TAG | TAGS | TBLPROPERTY | TBLPROPERTIES ) 
    [ IF EXISTS ] '<tag_name>'[ , '<tag_name>' ... ]
```
