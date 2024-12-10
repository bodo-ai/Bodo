# ARRAY_REMOVE_AT

`#!sql ARRAY_REMOVE_AT(array, pos)`

Given a source `array`, returns an array with the element at the specified position
`pos` removed. Returns `NULL` if `array` or `pos` is `NULL`.
Negative indexing is supported. No element is removed of the index `pos` is out of bound.
