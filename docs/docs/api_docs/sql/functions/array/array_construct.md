# ARRAY_CONSTRUCT


`#!sql ARRAY_CONSTRUCT(A, B, C, ...)`

Takes in a variable number of arguments and produces an array containing
all of those values (including any null values).

!!! note
    Snowflake allows any number of arguments (even zero arguments) of any
    type. BodoSQL currently requires 1+ arguments, and requires all arguments
    to be easily reconciled into a common type.

