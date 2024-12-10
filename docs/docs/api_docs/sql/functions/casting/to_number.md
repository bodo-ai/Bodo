# TO_NUMBER

`#!sql TO_NUMBER(EXPR [, PRECICION [, SCALE]])`

Converts an input expression to a fixed-point number with the specified precicion and scale.
Precicon and scale must be constant integer literals if provided. Precicion must be between
1 and 38. Scale must be between 0 and prec - 1.
Precicion and scale default to 38 and 0 if not provided. For `NULL` input,
the output is `NULL`.
