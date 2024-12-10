# NOT BETWEEN

The `#!sql BETWEEN` operator selects values within a given range. The
values can be numbers, text, or datetimes. The `#!sql BETWEEN` operator
is inclusive: begin and end values are included:

```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <COLUMN_NAME> BETWEEN <VALUE1> AND <VALUE2>
```

For example:

```sql
SELECT A FROM table1 WHERE A BETWEEN 10 AND 100
```

### Example Usage

```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df})
...    query = "SELECT name, balance FROM customers WHERE balance BETWEEN 1000 and 5000"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df})
...    query = "SELECT name, balance FROM customers WHERE balance NOT BETWEEN 100 and 10000"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "CUSTOMERID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "NAME": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "BALANCE": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g1(payment_df) # BETWEEN
            NAME  BALANCE
0  Deangelo Todd  1123.34
1   Nikolai Kent  2133.43
7  Shawn Winters  3489.25

>>>g2(payment_df) # NOT BETWEEN
                 NAME   BALANCE
2          Eden Heath     23.58
5    Weston Jefferson     68.34
6  Jonathon Middleton  12764.50
9      Darryl Rosales  25645.39
```
