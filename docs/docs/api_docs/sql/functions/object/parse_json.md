# PARSE_JSON

`#!sql PARSE_JSON(str)`

Takes in a string representing a json document and parses it to the
corresponding value as a variant. For example:

- `#!sql PARSE_JSON('42')` is equivalent to `#!sql TO_VARIANT(42)`

- `#!sql PARSE_JSON('{"A": 0, "B": 3.1}')` is equivalent to `#!sql TO_VARIANT({"A": 0, "B": 3.1})`

!!! note
Currently only supported under limited conditions where it is possible to rewrite
the call to `PARSE_JSON` as a sequence of Parse-Extract-Cast
operations, where the output of `PARSE_JSON` immediately has an extraction
operation like GET/GET_PATH called on it, and the result is casted to a
non-semi-structured type. For example, `#!sql PARSE_JSON(S):fizz::integer`
can be rewritten, as can `#!sql GET_PATH(TO_OBJECT(TO_ARRAY(PARSE_JSON(S))[0]), 'foo.bar')::varchar`.
