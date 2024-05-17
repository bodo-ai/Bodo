# OBJECT_PICK


`#!sql OBJECT_PICK(data, key1[, key2, ...])`

Takes in a column of object data and 1+ keys and returns the object data only
containing the keys specified. If a specified key is not present in `data`,
it is ignored.

!!! note: BodoSQL supports when the keys are passed in as string literals,
but only supports when they are passed in as columns of strings if the object
is a map instead of struct.



