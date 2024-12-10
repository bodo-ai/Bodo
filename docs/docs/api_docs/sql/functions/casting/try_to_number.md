# TRY_TO_NUMBER

`#!sql TRY_TO_NUMBER(EXPR [, PRECICION [, SCALE]])`

A special version of `#!sql TO_NUMBER` that performs
the same operation (i.e. converts an input expression to a fixed-point
number), but with error-handling support (i.e. if the conversion cannot be
performed, it returns a `NULL` value instead of raising an error).
