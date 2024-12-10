# OBJECT_DELETE

`#!sql OBJECT_DELETE(data, key1[, key2, ...])`

Takes in a column of JSON data and 1+ keys and returns the same JSON data but
with all of those keys removed. If a specified key is not present in
`data`, it is ignored.

!!! note: BodoSQL supports when the keys are passed in as string literals,
but only supports when they are passed in as columns of strings if the object
is a map instead of struct.
