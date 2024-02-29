# BASE64_ENCODE


`#!sql BASE64_ENCODE(msg[, max_line_length[, alphabet]])`

Encodes the `msg` string into a string using the base64 encoding scheme as if
it were binary data (or directly encodes binary data). If `#!sql max_line_length`
(default zero) is greater than zero, then newline characters will be inserted
after that many characters to effectively add "text wrapping". If `#!sql alphabet`
is provided, it specifies substitutes for the usual encoding characters for
index 62, index 63, and the padding character.
[See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/base64_encode).


