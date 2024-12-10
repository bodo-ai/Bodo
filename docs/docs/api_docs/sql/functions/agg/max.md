# MAX
`#!sql MAX`

Compute the maximum value in the column/group/window.
Supported on all non-semi-structured types.

Returns `#!sql NULL` if the input is all `#!sql NULL` or empty.

!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
