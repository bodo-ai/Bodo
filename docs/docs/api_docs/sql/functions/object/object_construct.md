# OBJECT_CONSTRUCT


`#!sql OBJECT_CONSTRUCT(key1, value1[, key2, value2, ...])`

The same as `#!sql OBJECT_CONSTRUCT_KEEP_NULL` except that for any rows where any input value
(e.g. `value1`, `value2`, ...) is null have that key-value pair dropped from the row's final JSON output.

!!! note
    BodoSQL only supports this function under narrow conditions where all of the values
    are either of the same type or of easily reconciled types.

[The full Snowflake specification](https://docs.snowflake.com/en/sql-reference/functions/object_construct.html).

BodoSQL supports the syntactic sugar `#!sql OBJECT_CONSTRUCT(*)`
which indicates that all columns should be used as key-value pairs, where
the column is the value and its column name is the key. For example, if we have
the table `T` as defined below:

| First    | Middle   | Last         |
|----------|----------|--------------|
| "George" | NULL     | "WASHINGTON" |
| "John"   | "Quincy" | "Adams"      |
| "Lyndon" | "Baines" | "Johnson"    |
| "James"  | NULL     | "Madison"    |

Then `SELECT OBJECT_CONSTRUCT(*) as name FROM T` returns the following table:

| name                                                      |
|-----------------------------------------------------------|
| {"First": "George", "Last": "Washington"}                 |
| {"First": "John", "Middle": "Quincy", "Last": "Adams"}    |
| {"First": "Lyndon", "Middle":"Baines", "Last": "Johnson"} |
| {"First": "Thomas", "Last": "Jefferson"}                  |


