# BOOLOR_AGG

`#!sql BOOLOR_AGG`

Compute the logical OR of the boolean value of every input
in a column/group/window, returning `#!sql NULL` if there are no non-`#!sql NULL` entries, otherwise
returning True if there is at least 1 non-zero entry. This is supported for
numeric and boolean types.
