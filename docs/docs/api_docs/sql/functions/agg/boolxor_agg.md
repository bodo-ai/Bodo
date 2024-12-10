# BOOLXOR_AGG

`#!sql BOOLXOR_AGG`

Returns `#!sql NULL` if there are no non-`#!sql NULL` entries, otherwise
returning True if exactly one non-`#!sql NULL` entry is also non-zero (this is
counterintuitive to how the logical XOR is normally thought of). This is
supported for numeric and boolean types.
