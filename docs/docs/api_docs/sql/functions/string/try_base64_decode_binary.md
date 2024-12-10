# TRY_BASE64_DECODE_BINARY

`#!sql TRY_BASE64_DECODE_BINARY(msg[, alphabet])`

Equivalent to `#!sql BASE64_DECODE_BINARY` except that it will return null instead of raising
an exception if the string is malformed in any way.
