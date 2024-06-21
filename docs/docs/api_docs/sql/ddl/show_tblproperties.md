# SHOW TBLPROPERTIES

Returns the value of a table property given an optional value for a property key. If no key is specified then all the properties are returned.

!!! note
    This operation is an Iceberg-only operation.

### Syntax

```sql
SHOW ( TBLPROPERTIES | PROPERTIES | TAGS ) <table_identifier> [ ('property_key') ] 
```

### Examples

```sql
SHOW TBLPROPERTIES my_table;

  +---------------------+----------+
  |key                  |value     |
  +---------------------+----------+
  |property_1           |value_1   |
  |property_2           |value_2   |
  |property_3           |value_3   |
  +---------------------+----------+

SHOW TBLPROPERTIES my_table ('property_1');

  +----------+
  |value     |
  +----------+
  |value_1   |
  +----------+
```

