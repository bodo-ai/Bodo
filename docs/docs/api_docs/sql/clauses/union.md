# UNION



The `#!sql UNION` operator is used to combine the result-set of two `#!sql SELECT`
statements:
```sql
SELECT <COLUMN_NAMES> FROM <TABLE1>
UNION
SELECT <COLUMN_NAMES> FROM <TABLE2>
```
Each `#!sql SELECT` statement within the `#!sql UNION` clause must have the same
number of columns. The columns must also have similar data types.
The output of the `#!sql UNION` is the set of rows which are present in
either of the input `#!sql SELECT` statements.

The `#!sql UNION` operator selects only the distinct values from the
inputs by default. To allow duplicate values, use `#!sql UNION ALL`:

```sql
SELECT <COLUMN_NAMES> FROM <TABLE1>
UNION ALL
SELECT <COLUMN_NAMES> FROM <TABLE2>
```

### Example Usage

```py
>>>@bodo.jit
... def g1(df):
...    bc = bodosql.BodoSQLContext({"CUSTOMERS":df1, "PAYMENTS":df2})
...    query = "SELECT name, \"paymentType\" FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE \"paymentType\" in ('WIRE')
...             UNION SELECT name, paymentType FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE balance < 1000"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"customers":df1, "payments":df2})
...    query = "SELECT name, \"paymentType\" FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE \"paymentType\" in ('WIRE')
...             UNION ALL SELECT name, paymentType FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE balance < 1000"
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

>>>g1(customer_df, payment_df) # UNION
           NAME paymentType  BALANCE
0  Demetrius Chavez        WIRE   943.43
0        Eden Heath        AMEX    23.58

>>>g2(customer_df, payment_df) # UNION ALL
            NAME paymentType  BALANCE
0  Demetrius Chavez        WIRE   943.43
0        Eden Heath        AMEX    23.58
1  Demetrius Chavez        WIRE   943.43
```

