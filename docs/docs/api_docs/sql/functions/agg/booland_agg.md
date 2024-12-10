# BOOLAND_AGG
`#!sql BOOLAND_AGG`

Compute the logical AND of the boolean value of every input
in a column/group/window, returning `#!sql NULL` if there are no non-`#!sql NULL` entries, otherwise
returning True if all non-`#!sql NULL` entries are also non-zero. This is supported for
numeric and boolean types.


