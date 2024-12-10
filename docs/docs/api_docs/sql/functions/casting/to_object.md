# TO_OBJECT

`#!sql TO_OBJECT(EXPR)`

If the input is an object type or a variant containing an object type, returns the input
unmodified (except that its type is now `#!sql OBJECT` if it was `#!sql VARIANT`). For
all other types, raises an error. Returns `NULL` for `NULL` or a JSON null input.
