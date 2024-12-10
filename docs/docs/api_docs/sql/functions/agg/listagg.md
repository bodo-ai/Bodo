# LISTAGG

`#!sql LISTAGG(str_col[, delimeter]) [WITHIN GROUP (ORDER BY order_col)]`

Concatenates all the strings in `str_col` within each group into a single
string seperated by the characters in the string `delimiter`. If no delimiter
is provided, an empty string is used by default.

Optionally allows using a `WITHIN GROUP` clause to specify how the strings should
be ordered before being concatenated. If no clause is specified, then the ordering
is unpredictable.

Returns `#!sql ''` if the input is all `#!sql NULL` or empty.
