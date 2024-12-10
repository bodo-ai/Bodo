# LEAD

`#!sql LEAD(COLUMN_EXPRESSION, [, N[, FILL_VALUE]])`

Equivalent to `#!sql LEAD(COLUMN_EXPRESSION, -N, FILL_VALUE)`,
in other words, returns the row following the current row by N.
