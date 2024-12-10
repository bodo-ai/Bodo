# TRY_HEX_DECODE_STRING

`#!sql TRY_HEX_DECODE_STRING(msg)`

Equivalent to `#!sql HEX_DECODE_STRING` except that it will return null instead of raising
an exception if the string is malformed in any way.
