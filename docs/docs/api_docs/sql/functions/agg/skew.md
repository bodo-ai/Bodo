# SKEW

`#!sql SKEW`

Compute the skew of a column/group/window or `NULL` if the window contains fewer
than 3 non-`NULL` entries. Supported on numeric types.

Returns `#!sql NULL` if the input is all `#!sql NULL` or empty.
