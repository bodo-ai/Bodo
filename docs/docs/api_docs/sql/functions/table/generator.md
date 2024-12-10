# GENERATOR

`#!sql GENERATOR([ROWCOUNT=>count][, TIMELIMIT=>sec])`

Generates a table with a certain number of rows, specified by the `ROWCOUNT` argument.
Currently only supports when the `ROW_COUNT` argument is provided and when it is a
non-negative integer. Does not support when the `TIMELIMIT` argument is provided, neither
argument is provided, or both are provided.
