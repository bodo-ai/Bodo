# HASH

`#!sql HASH(A, B, C, ...)`

Takes in a variable number of arguments of any type and returns a hash
value that considers the values in each column. The hash function is
deterministic across multiple ranks or multiple sessions.

Also supports the syntactic sugar forms `#!sql HASH(*)` and `#!sql HASH(T.*)`
as shortcuts for referencing all of the columns in a table, or multiple tables.
For example, if `#!sql T1` has columns `A` and `B`, and `T2` has columns
`A`, `E` and `I`, then the following query:

`#!sql SELECT HASH(*), HASH(T1.*) FROM T1 INNER JOIN T2 ON T1.A=T2.I`

Would be syntactic sugar for the following:

`#!sql SELECT HASH(T1.A, T1.B, T2.A, T2.E, T2.I), HASH(T1.A, T1.B) FROM T1 INNER JOIN T2 ON T1.A=T2.I`
