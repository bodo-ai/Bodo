# LIMIT

BodoSQL supports the `#!sql LIMIT` keyword to select a limited number
of rows. This keyword can optionally include an offset:

```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <CONDITION>
LIMIT <LIMIT_NUMBER> OFFSET <OFFSET_NUMBER>
```

For Example:

```sql
SELECT A FROM table1 LIMIT 5

SELECT B FROM table2 LIMIT 8 OFFSET 3
```

Specifying a limit and offset can be also be written as:

```sql
LIMIT <OFFSET_NUMBER>, <LIMIT_NUMBER>
```

For Example:

```sql
SELECT B FROM table2 LIMIT 3, 8
```

### Example Usage

```py
>>>@bodo.jit
... def g1(df):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df})
...    query = "SELECT name FROM customers LIMIT 4"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df})
...    query = "SELECT name FROM customers LIMIT 4 OFFSET 2"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "CUSTOMERID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "NAME": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "BALANCE": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g1(customers_df) # LIMIT 4
               NAME
0     Deangelo Todd
1      Nikolai Kent
2        Eden Heath
3  Taliyah Martinez

>>>g2(customers_df) # LIMIT 4 OFFSET 2
               NAME
2        Eden Heath
3  Taliyah Martinez
4  Demetrius Chavez
5  Weston Jefferson
```
