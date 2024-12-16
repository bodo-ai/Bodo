# FLATTEN


`#!sql FLATTEN([INPUT=>]expr[, PATH=>path_epxr][, OUTER=>outer_expr][, RECURSIVE=>recursive_expr][, MODE=>mode_epxr])`

Takes in a column of semi-structured data and produces a table by
"exploding" the data into multiple rows, producing the following
columns:

- `#!sql SEQ`: not currently supported by BodoSQL.
- `#!sql KEY`: the individual values from the json data.
- `#!sql PATH`: not currently supported by BodoSQL.
- `#!sql INDEX`: the index within the array that the value came from.
- `#!sql VALUE`: the individual values from the array or json data.
- `#!sql THIS`: a copy of the input data.

The function has the following named arguments:

- `#!sql INPUT` (required): the expression of semi-structured data to flatten. Also allowed to be passed in as a positional argument without the `INPUT` keyword.
- `#!sql PATH` (optional): a constant expression referencing how to access the semi-structured data to flatten from the input expression. BodoSQL currently only supports when this argument is omitted or is an empty string (indicating that the expression itself is the array to flatten).
- `#!sql OUTER` (optional): a boolean indicating if a row should be generated even if the input data is an
empty/null array/struct/map. The default is false. If provided, the `KEY`, `PATH`, `INDEX` and `VALUE` outputs will be null in the generated row.
- `#!sql RECURSIVE` (optional): a boolean indicating if flattening should occur recursively, as opposed to just on the data referenced by `PATH`. BodoSQL currently only supports when this argument is omitted or is false (which is the default).
- `#!sql MODE` (optional): a string literal that can be either `'OBJECT'`, `'ARRAY'` or `'BOTH'`, indicating what type of flattening rule should be done. BodoSQL currently only supports when this argument is omitted or is `'BOTH'` (which is the default).

!!! note
    BodoSQL supports the input argument being an array, json or variant
    so long as the values are of the same type (with limited support for
    JSON when the values are also JSON).


Below is an example of a query using the `#!sql FLATTEN` function with the
`#!sql LATERAL` keyword to explode an array column while also
replicating another column.

```sql
SELECT id, lat.index as idx, lat.value as val FROM table1, lateral flatten(tags) lat
```

If the input data was as follows:

| id | tags                      |
|----|---------------------------|
| 10 | ["A", "B"]                |
| 16 | []                        |
| 72 | ["C", "A", "B", "D", "C"] |

Then the query would produce the following data:

| id | idx | val |
|----|-----|-----|
| 10 | 0   | "A" |
| 10 | 1   | "B" |
| 72 | 0   | "C" |
| 72 | 1   | "A" |
| 72 | 2   | "B" |
| 72 | 3   | "D" |
| 72 | 4   | "C" |

Below is an example of a query using the `#!sql FLATTEN` function with the
`#!sql LATERAL` keyword to explode an JSON column while also
replicating another column.

```sql
SELECT id, lat.key as key, lat.value as val FROM table1, lateral flatten(attributes) lat
```

If the input data was as follows:

| id | attributes       |
|----|------------------|
| 42 | {"A": 0}         |
| 50 | {}               |
| 64 | {"B": 1, "C": 2} |

Then the query would produce the following data:

| id | key | value |
|----|-----|-------|
| 42 | "A" | 0     |
| 64 | "B" | 1     |
| 64 | "C" | 2     |


