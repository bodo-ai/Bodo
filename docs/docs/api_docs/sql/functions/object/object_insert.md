# OBJECT_INSERT

`#!sql OBJECT_INSERT(data, key, value[, update])`

Takes a columns of JSON data, a column of string keys, and a columns of
values and inserts the keys and values into the data. If the key is already
present in the data, an error will be thrown, unless an additional argument
(`update`) of type boolean is supplied, which will update existing keys to
hold the new value only if the value is true.
