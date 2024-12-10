# WHERE

The `#!sql WHERE` clause on columns can be used to filter records that
satisfy specific conditions:

```sql
SELECT <COLUMN_NAMES> FROM <TABLE_NAME> WHERE <CONDITION>
```

For Example:

```sql
SELECT A FROM table1 WHERE B > 4
```

### Example Usage

```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df})
...    query = "SELECT name FROM customers WHERE balance 3000"
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
3    Taliyah Martinez
6  Jonathon Middleton
7       Shawn Winters
9      Darryl Rosales
```
