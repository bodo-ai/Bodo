# CONDITIONAL_CHANGE_EVENT

`#!sql CONDITIONAL_CHANGE_EVENT(COLUMN_EXPRESSION)`

Computes a counter within each partition that starts at zero and increases by 1 each
time the value inside the window changes. `NULL` does not count as a new/changed value.
`#!sql ORDER BY` is required for this function.
