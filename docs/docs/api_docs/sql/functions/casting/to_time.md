# TO_TIME


`#!sql TO_TIME(EXPR)`

Converts an input expression to a `TIME` type. The input can be one of
the following:

- `#!sql TO_TIME(timestamp_expr)` extracts the time component from a timestamp.
- `#!sql TO_TIME(string_expr)` if the string is in date format (e.g. `"12:30:15"`)
then it is convrted to a corresponding time.
- `#!sql TO_TIME(string_expr, format_expr)` uses the format string to specify how to parse the
string expression as a time. Uses the format string rules [as specified by Snowflake](https://docs.snowflake.com/en/sql-reference/functions-conversion#label-date-time-format-conversion).
- If the input is `NULL`, outputs `NULL`

Raises an error if the input expression does not match one of these formats.

