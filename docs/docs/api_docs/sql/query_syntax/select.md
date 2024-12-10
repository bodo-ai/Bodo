# SELECT

The `#!sql SELECT` statement is used to select data in the form of
columns. The data returned from BodoSQL is stored in a dataframe.

```sql
SELECT <COLUMN_NAMES> FROM <TABLE_NAME>
```

For Instance:

```sql
SELECT A FROM customers
```

### Example Usage:

```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df})
...    query = "SELECT name FROM customers"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "CUSTOMERID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "NAME": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "BALANCE": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g(customers_df)
                NAME
0       Deangelo Todd
1        Nikolai Kent
2          Eden Heath
3    Taliyah Martinez
4    Demetrius Chavez
5    Weston Jefferson
6  Jonathon Middleton
7       Shawn Winters
8    Keely Hutchinson
9      Darryl Rosales
```

The `#!sql SELECT` also has some special syntactic forms. The `#!sql *` term is
used as a shortcut for specifying all columns. The clause `#!sql * EXCLUDE col`
or `#!sql * EXCLUDE (col1, col2, col3...)` is a shortcut for specifying every
column except the ones after the EXCLUDE keyword.

For example, suppose we have a table The `#!sql T` with columns named The `#!sql A`, `#!sql B`,
`#!sql C`, `#!sql D`, `#!sql E`. Consider the following queries

```sql
SELECT * FROM T

SELECT * EXCLUDE (A, E) FROM T
```

These two are syntactic sugar for the following:

```sql
SELECT A, B, C, D, E FROM T

SELECT B, C, D FROM T
```
