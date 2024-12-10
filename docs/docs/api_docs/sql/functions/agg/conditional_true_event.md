# CONDITIONAL_TRUE_EVENT

`#!sql CONDITIONAL_TRUE_EVENT(BOOLEAN_COLUMN_EXPRESSION)`

Computes a counter within each partition that starts at zero and increases by 1 each
time the boolean column's value is `true`. `#!sql ORDER BY` is required for this function.
