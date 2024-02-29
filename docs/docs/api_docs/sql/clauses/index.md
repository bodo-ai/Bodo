# Clauses

We currently support the following SQL query statements and clauses with
BodoSQL, and are continuously adding support towards completeness. Note
that BodoSQL ignores casing of keywords, and column and table names,
except for the final output column name. Therefore,
`#!sql select a from table1` is treated the same as `#!sql SELECT A FROM Table1`,
except for the names of the final output columns (`a` vs `A`).

- [Aliasing][aliasing]
- [`#!sql CASE`][case]
- [`#!sql CAST`][cast]
- [`#!sql GREATEST`][greatest]
- [`#!sql GROUP BY`][group-by]
- [`#!sql HAVING`][having]
- [`#!sql INDEX`][index]
- [`#!sql ::`][infix]
- [`#!sql INTERSECT`][intersect]
- [`#!sql JOIN`][join]
- [`#!sql LEAST`][least]
- [`#!sql LIKE`][like]
- [`#!sql LIMIT`][limit]
- [`#!sql NATURAL JOIN`][natural-join]
- [`#!sql NOT BETWEEN`][not-between]
- [`#!sql NOT IN`][not-in]
- [`#!sql ORDER BY`][order-by]
- [`#!sql PIVOT`][pivot]
- [`#!sql QUALIFY`][qualify]
- [`#!sql SELECT`][select]
- [`#!sql SELECT DISTINCT`][select-distinct]
