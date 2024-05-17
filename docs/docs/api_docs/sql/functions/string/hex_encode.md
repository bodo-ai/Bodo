# HEX_ENCODE


`#!sql HEX_ENCODE(msg[, case])`

Encodes the `msg` string into a string using the hex encoding scheme as if
it were binary data (or directly encodes binary data). If `#!sql case`
(default one) is zero then the alphabetical hex characters are lowercase,
if it is one then they are uppercase.
[See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/hex_encode).


