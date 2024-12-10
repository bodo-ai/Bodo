# TRY_BASE64_DECODE_STRING

`#!sql TRY_BASE64_DECODE_STRING(msg[, alphabet])`

Equivalent to `#!sql BASE64_DECODE_STRING` except that it will return null instead of raising
an exception if the string is malformed in any way.
