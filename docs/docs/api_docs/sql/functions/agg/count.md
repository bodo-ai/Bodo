# COUNT
`#!sql COUNT`

Count the number of non-null elements in the column/group/window.
Supported on all types. If used with the syntax `#!sql COUNT(*)`
returns the total number of rows instead of non-null rows.

!!! note
    When used as a window function with an `#!sql ORDER BY` clause but no window frame, `#!sql ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is used by default.
