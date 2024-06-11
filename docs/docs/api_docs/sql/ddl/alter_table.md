# ALTER TABLE

Modifies the properties, columns, or constraints for an existing table from the current/specified schema.

See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/sql/alter-table) and [Iceberg documentation](https://iceberg.apache.org/docs/nightly/spark-ddl/#alter-table) for more details.

Currently, BodoSQL only supports the following operations:

## Renaming a table
```sql
ALTER TABLE [ IF EXISTS ] <name> RENAME TO <new_table_name>
```

## Adding columns

```sql
ALTER TABLE [ IF EXISTS ] <name> 
    ADD [ COLUMN ] [ IF NOT EXISTS ] <new_column_name> <column_datatype>
```

??? warning
    - Note that BodoSQL does not yet support `ADD COLUMN` for Snowflake.

    - Only a subset of Iceberg types are supported for `ADD COLUMN`. The syntax corresponding to the Iceberg types are as follows:

        | Syntax                                                    | Iceberg Type   |
        |:----------------------------------------------------------|:---------------|
        | DECIMAL, NUMERIC                                          | decimal(38, 0) |
        | NUMBER(P, S), DECIMAL(P, S)                               | decimal(p, s)  | 
        | INT, INTEGER, SMALLINT, TINYINT, BYTEINT                  | int            |
        | BIGINT                                                    | long           |
        | FLOAT, FLOAT4, FLOAT8                                     | float          |
        | DOUBLE, DOUBLE PRECISION, REAL                            | double         |
        | VARCHAR, CHAR, CHARACTER, STRING, TEXT, BINARY, VARBINARY | string         |
        | BOOLEAN                                                   | boolean        |
        | DATE                                                      | date           |
        | TIME                                                      | time           |
        | DATETIME, TIMESTAMP, TIMESTAMP_NTZ                        | timestamp      |

        Note that adding nested types such as `struct<x: double, y: double>` is not supported yet. As such, column names including periods are disallowed in order to prevent ambiguity.

## Dropping columns
```sql
ALTER TABLE [ IF EXISTS ] <name> 
    DROP [ COLUMN ] [ IF EXISTS ] <column_name> [ , <column_name>, ...]
```

??? note
    -  Note that BodoSQL does not yet support `DROP COLUMN` for Snowflake.

    -  `DROP COLUMN` can be used to drop nested columns and fields of structs.
    
        To do so, use `.` separated field names:

        ```sql
        -- Example
        ALTER TABLE tblname DROP COLUMN colname.fieldname
        ```

        Multiple nested columns are also supported:
        
        ```sql
        -- Example
        ALTER TABLE tblname DROP COLUMN colname.structname.fieldname
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

## Setting / unsetting table comments

This operation functions as an alias for `ALTER TABLE SET PROPERTY COMMENT='comment'`.

!!! note
    This operation is currently only supported for Iceberg.

```sql
ALTER TABLE [ IF EXISTS ] <name> SET COMMENT '<comment>'
```

```sql
ALTER TABLE [ IF EXISTS ] <name> UNSET COMMENT
```