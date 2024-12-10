# OBJECT_CONSTRUCT_KEEP_NULL

`#!sql OBJECT_CONSTRUCT_KEEP_NULL(key1, value1[, key2, value2, ...])`

Takes in a variable number of key-value pairs and combines them
into JSON data. BodoSQL currently requires all `key` arguments to
be string literals.

[The full Snowflake specification](https://docs.snowflake.com/en/sql-reference/functions/object_construct_keep_null.html).

BodoSQL supports the syntactic sugar `#!sql OBJECT_CONSTRUCT_KEEP_NULL(*)`
which indicates that all columns should be used as key-value pairs, where
the column is the value and its column name is the key. For example, if we have
the table `T` as defined below:

| First | Middle | Last |
|----------|----------|--------------|
| "George" | NULL | "WASHINGTON" |
| "John" | "Quincy" | "Adams" |
| "Lyndon" | "Baines" | "Johnson" |
| "James" | NULL | "Madison" |

Then `SELECT OBJECT_CONSTRUCT_KEEP_NULL(*) as name FROM T` returns the following table:

| name |
|-----------------------------------------------------------|
| {"First": "George", "Middle": NULL, "Last": "Washington"} |
| {"First": "John", "Middle": "Quincy", "Last": "Adams"} |
| {"First": "Lyndon", "Middle":"Baines", "Last": "Johnson"} |
| {"First": "Thomas", "Middle": NULL, "Last": "Jefferson"} |
