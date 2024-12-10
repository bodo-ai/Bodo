# SPLIT_TO_TABLE


`#!sql SPLIT_TO_TABLE(str, delim)`

Takes in a string column and a delimeter and produces a table by
"exploding" the string into multiple rows based on the delimeter,
producing the following columns:

- `#!sql SEQ`: not currently supported by BodoSQL.
- `#!sql INDEX`: which index in the splitted string did the current seciton come from.
- `#!sql VALUE`: the current section of the splitted string.

!!! note
    Currently, BodoSQL supports this function as an alias
    for `#!sql FLATTEN(SPLIT(str, delim))`.

Below is an example of a query using the `#!sql SPLIT_TO_TABLE` function with the
`#!sql LATERAL` keyword to explode an string column while also
replicating another column.

```sql
SELECT id, lat.index as idx, lat.value as val FROM table1, lateral split_to_table(colors, ' ') lat
```

If the input data was as follows:

| id | colors              |
|----|---------------------|
| 50 | "red orange yellow" |
| 75 | "green blue"        |

Then the query would produce the following data:

| id | idx | val      |
|----|-----|----------|
| 50 | 0   | "red"    |
| 50 | 1   | "orange" |
| 50 | 2   | "yellow" |
| 75 | 0   | "green"  |
| 75 | 1   | "blue"   |


