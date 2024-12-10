# TO_ARRAY


`#!sql TO_ARRAY(EXPR)`

Converts the input expression to a single-element array containing this value. If the input
is an ARRAY, or VARIANT containing an array value, the result is unchanged. Returns `NULL`
for `NULL` or a JSON null input.

