# JOIN



A `#!sql JOIN` clause is used to combine rows from two or more tables,
based on a related column between them:
```sql
SELECT <COLUMN_NAMES>
  FROM <LEFT_TABLE_NAME>
  <JOIN_TYPE> <RIGHT_TABLE_NAME>
  ON <LEFT_TABLE_COLUMN_NAME> OP <RIGHT_TABLE_COLUMN_NAME>
```
For example:
```sql
SELECT table1.A, table1.B FROM table1 JOIN table2 on table1.A = table2.C
```
Here are the different types of the joins in SQL:

-   `#!sql (INNER) JOIN`: returns records that have matching values in
both tables
-   `#!sql LEFT (OUTER) JOIN`: returns all records from the left table,
and the matched records from the right table
-   `#!sql RIGHT (OUTER) JOIN`: returns all records from the right
table, and the matched records from the left table
-   `#!sql FULL (OUTER) JOIN`: returns all records when there is a match
in either left or right table

### Example Usage

```py
>>>@bodo.jit
... def g1(df1, df2):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df1, "PAYMENTS":df2})
...    query = "SELECT name, \"paymentType\" FROM customers JOIN payments ON customers.customerID = payments.customerID"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df1, df2):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df1, "PAYMENTS":df2})
...    query = "SELECT name, paymentType FROM customers FULL JOIN payments ON customers.customerID = payments.customerID"
...    res = bc.sql(query)
...    return res

>>>customer_df = pd.DataFrame({
...    "CUSTOMERID": [0, 2, 4, 5, 7,],
...    "NAME": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez","Demetrius Chavez",],
...    "ADDRESS": ["223 Iroquois LanenWest New York, NJ 07093","37 Depot StreetnTaunton, MA 02780",
...                "639 Maple St.nNorth Kingstown, RI 02852","93 Bowman Rd.nChester, PA 19013",
...                "513 Manchester Ave.nWindsor, CT 06095",],
...    "BALANCE": [1123.34, 2133.43, 23.58, 8345.15, 943.43,]
... })
>>>payment_df = pd.DataFrame({
...     "CUSTOMERID": [0, 1, 4, 6, 7],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE",],
... })

>>>g1(customer_df, payment_df) # INNER JOIN
               NAME paymentType
0     Deangelo Todd        VISA
1        Eden Heath        AMEX
2  Demetrius Chavez        WIRE

>>>g2(customer_df, payment_df) # OUTER JOIN
               NAME paymentType
0     Deangelo Todd        VISA
1      Nikolai Kent         NaN
2        Eden Heath        AMEX
3  Taliyah Martinez         NaN
4  Demetrius Chavez        WIRE
5               NaN        VISA
6               NaN        VISA
```

