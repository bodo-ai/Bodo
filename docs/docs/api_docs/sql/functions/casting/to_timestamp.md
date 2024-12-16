# TO_TIMESTAMP


`#!sql TO_TIMESTAMP(EXPR)`

Converts an input expression to a `TIMESTAMP` type without a timezone. The input can be one of
the following:

- `#!sql TO_TIMESTAMP(date_expr)` upcasts a `DATE` to a `TIMESTAMP`.
- `#!sql TO_TIMESTAMP(integer)` creates a timestamp using the integer as the number of
seconds/milliseconds/microseconds/nanoseconds since `1970-01-1`. Which unit it is interpreted
as depends on the magnitude of the number, in accordance with [the semantics used by Snowflake](https://docs.snowflake.com/en/sql-reference/functions/to_date#usage-notes).
- `#!sql TO_TIMESTAMP(integer, scale)` the same as the integer case except that the scale provided specifes which
unit is used. THe scale can be an integer constant between 0 and 9, where 0 means seconds and 9 means nanoseconds.
- `#!sql TO_TIMESTAMP(string_expr)` if the string is in timestamp format (e.g. `"1999-12-31 23:59:30"`)
then it is convrted to a corresponding timestamp. If the string represents an integer
(e.g. `"123456"`) then it uses the same rule as the corresponding input integer.
- `#!sql TO_TIMESTAMP(string_expr, format_expr)` uses the format string to specify how to parse the
string expression as a timestamp. Uses the format string rules [as specified by Snowflake](https://docs.snowflake.com/en/sql-reference/functions-conversion#label-date-time-format-conversion).
- `#!sql TO_TIMESTAMP(timestamp_exr)` returns a timestamp expression representing the same moment in time,
but changing the timezone or offset if necessary to be timezone-naive.
- If the input is `NULL`, outputs `NULL`

Raises an error if the input expression does not match one of these formats.

