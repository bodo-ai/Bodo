# BASE64_DECODE_STRING


`#!sql BASE64_DECODE_STRING(msg[, alphabet])`

Reverses the process of calling `#!sql BASE64_ENCODE` on a string with the given alphabet,
ignoring any newline characters produced by the `#!sql max_line_length` argument. Raises an
exception if the string is malformed in any way.
[See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/base64_decode_string).


