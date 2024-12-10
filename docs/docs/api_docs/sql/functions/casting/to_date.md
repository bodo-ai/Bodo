# TO_DATE

`#!sql TO_DATE(EXPR)`

Converts an input expression to a `DATE` type. The input can be one of
the following:

- `#!sql TO_DATE(timestamp_expr)` truncates the timestamp to its date value.
- `#!sql TO_DATE(timestamptz_expr)` truncates the TIMESTAMPTZ to its date value based on it's local timestamp (not UTC).
- `#!sql TO_DATE(string_expr)` if the string is in date format (e.g. `"1999-01-01"`)
  then it is convrted to a corresponding date. If the string represents an integer
  (e.g. `"123456"`) then it is interpreted as the number of seconds/milliseconds/microseconds/nanoseconds
  since `1970-01-1`. Which unit it is interpreted as depends on the magnitude of the number,
  in accordance with [the semantics used by Snowflake](https://docs.snowflake.com/en/sql-reference/functions/to_date#usage-notes).
- `#!sql TO_DATE(string_expr, format_expr)` uses the format string to specify how to parse the
  string expression as a date. Uses the format string rules [as specified by Snowflake](https://docs.snowflake.com/en/sql-reference/functions-conversion#label-date-time-format-conversion).
- If the input is `NULL`, outputs `NULL`.

Raises an error if the input expression does not match one of these formats.
