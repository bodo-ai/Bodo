# INSERT


`#!sql INSERT(str1, pos, len, str2)`

Inserts `str2` into `str1` at position `pos` (1-indexed), replacing
the first `len` characters after `pos` in the process. If `len` is zero,
inserts `str2` into `str1` without deleting any characters. If `pos` is one,
prepends `str2` to `str1`. If `pos` is larger than the length of `str1`, appends
`str2` to `str1`.

!!! note
    Behavior when `pos` or `len` are negative is not well-defined at this time.


