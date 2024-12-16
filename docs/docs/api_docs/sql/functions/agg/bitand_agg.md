# BITAND_AGG
`#!sql BITAND_AGG`

Compute the bitwise AND of every input
in a column/group/window, returning `#!sql NULL` if there are no non-`#!sql NULL` entries.
Accepts floating point values, integer values, and strings. Strings are interpreted
directly as numbers, converting to 64-bit floating point numbers.


