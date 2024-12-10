# OBJECT_AGG

`#!sql OBJECT_AGG(K, V)`

Combines the data from columns `K` and `V` into a JSON object where the rows of
column `K` are the field names and the rows of column `V` are the values. Any
row where `K` or `V` is `NULL` is not included in the final object. If the group
is empty or all the rows are not included, an empty object is returned.
