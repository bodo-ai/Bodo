# COALESCE


`#!sql COALESCE(A, B, C, ...)`

Returns the first non-`NULL` argument, or `NULL` if no non-`NULL`
argument is found. Requires at least two arguments. If
Arguments do not have the same type, BodoSQL will attempt
to cast them to a common data type, which is currently
undefined behavior.


