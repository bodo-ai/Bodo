# ARRAY_POSITION

`#!sql ARRAY_POSITION(elem, arr)`

Returns the index of the first occurrence of `elem` in `arr` (using zero indexing), or
`NULL` if there are no occurrences. The input `elem` can be `NULL`, in which case the funciton
will look for the first `NULL` in the array input.
