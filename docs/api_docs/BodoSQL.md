BodoSQL {#bodosql}
========

BodoSQL provides high performance and scalable SQL query execution
using Bodo's HPC capabilities and optimizations. It also provides
native Python/SQL integration as well as SQL to Pandas conversion for
the first time.

## Getting Started

### Installation

Install BodoSQL using:

```shell
conda install bodosql -c bodo.ai -c conda-forge
```

### Using BodoSQL

The example below demonstrates using BodoSQL in Python programs. It
loads data into a dataframe, runs a SQL query on the data, and runs
Python/Pandas code on query results:

```py
import pandas as pd
import bodo
import bodosql

@bodo.jit
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"table1": df1})
    df2 = bc.sql("SELECT A FROM table1 WHERE B > 4")
    print(df2.A.sum())

f("my_data.pq")
```

This program is fully type checked, optimized and parallelized by Bodo
end-to-end. `BodoSQLContext` creates a SQL environment with tables
created from dataframes. `BodoSQLContext.sql()` runs a SQL query and
returns the results as a dataframe. `BodoSQLContext` can be used outside
Bodo JIT functions if necessary as well.

You can run this example by creating `my_data.pq`:

```py
import pandas as pd
import numpy as np

NUM_GROUPS = 30
NUM_ROWS = 20_000_000
df = pd.DataFrame({
    "A": np.arange(NUM_ROWS) % NUM_GROUPS,
    "B": np.arange(NUM_ROWS)
})
df.to_parquet("my_data.pq")
```

To run the example, save it in a file called `example.py` and run it using `mpiexec`, e.g.:

```console
mpiexec -n 8 python example.py
```

## Aliasing

In all but the most trivial cases, BodoSQL generates internal names to
avoid conflicts in the intermediate dataframes. By default, BodoSQL
does not rename the columns for the final output of a query using a
consistent approach. For example the query:

```sql
bc.sql("SELECT SUM(A) FROM table1 WHERE B > 4")
```
Results in an output column named `$EXPR0`. To reliably reference this
column later in your code, we highly recommend using aliases for all
columns that are the final outputs of a query, such as:

```py
bc.sql("SELECT SUM(A) as sum_col FROM table1 WHERE B > 4")
```

!!! note
     BodoSQL supports using aliases generated in `#!sql SELECT` inside
    `#!sql GROUP BY` and `#!sql HAVING` in the same query, but you cannot do so with
    `#!sql WHERE`.

## Supported Operations

We currently support the following SQL query statements and clauses with
BodoSQL, and are continuously adding support towards completeness. Note
that BodoSQL ignores casing of keywords, and column and table names,
except for the final output column name. Therefore,
`#!sql select a from table1` is treated the same as `#!sql SELECT A FROM Table1`,
except for the names of the final output columns (`a` vs `A`).

### SELECT

The `#!sql SELECT` statement is used to select data in the form of
columns. The data returned from BodoSQL is stored in a dataframe.

```sql
SELECT <COLUMN_NAMES> FROM <TABLE_NAME>
```

For Instance:

```sql
SELECT A FROM customers
```

***Example Usage***:

```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"customers":df})
...    query = "SELECT name FROM customers"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "customerID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g(customers_df)
                name
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

### SELECT DISTINCT

The `#!sql SELECT DISTINCT` statement is used to return only distinct
(different) values:

```sql
SELECT DISTINCT <COLUMN_NAMES> FROM <TABLE_NAME>
```

`#!sql DISTINCT` can be used in a SELECT statement or inside an
aggregate function. For example:

```sql
SELECT DISTINCT A FROM table1

SELECT COUNT DISTINCT A FROM table1
```

***Example Usage***
```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"payments":df})
...    query = "SELECT DISTINCT paymentType FROM payments"
...    res = bc.sql(query)
...    return res

>>>payment_df = pd.DataFrame({
...     "customerID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE", "VISA", "VISA", "WIRE", "VISA", "AMEX"],
... })

>>>g(payment_df) # inside SELECT
paymentType
0        VISA
2        AMEX
4        WIRE

>>>def g(df):
...    bc = bodosql.BodoSQLContext({"payments":df})
...    query = "SELECT COUNT(DISTINCT paymentType) as num_payment_types FROM payments"
...    res = bc.sql(query)
...    return res

>>>g(payment_df) # inside aggregate
num_payment_types
0          3
```

### WHERE

The `#!sql WHERE` clause on columns can be used to filter records that
satisfy specific conditions:

```sql
SELECT <COLUMN_NAMES> FROM <TABLE_NAME> WHERE <CONDITION>
```

For Example:
```sql
SELECT A FROM table1 WHERE B > 4
```

***Example Usage***
```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"customers":df})
...    query = "SELECT name FROM customers WHERE balance 3000"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "customerID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g(customers_df)
                name
3    Taliyah Martinez
6  Jonathon Middleton
7       Shawn Winters
9      Darryl Rosales
```

### ORDER BY

The `#!sql ORDER BY` keyword sorts the resulting DataFrame in ascending
or descending order. By default, it sorts the records in ascending order.
NULLs are sorted in accordance with the optional `#!sql NULLS FIRST` or
`#!sql NULLS LAST` keywords.

If the null ordering is not provided, then the default ordering depends
on if the column is ascending or descending. For ascending order, by
default NULL values are returned at the end, while for descending order
nulls are returned at the front. If the order of nulls matter we strongly
recommend explicitly providing either `#!sql NULLS FIRST` or
`#!sql NULLS LAST`.

```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
ORDER BY <ORDERED_COLUMN_NAMES> [ASC|DESC] [NULLS FIRST|LAST]
```

For Example:
```sql
SELECT A, B FROM table1 ORDER BY B, A DESC NULLS LAST
```

***Example Usage***


```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"customers":df})
...    query = "SELECT name, balance FROM customers ORDER BY balance"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "customerID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g(customers_df)
                name   balance
2          Eden Heath     23.58
5    Weston Jefferson     68.34
8    Keely Hutchinson    654.24
4    Demetrius Chavez    943.43
0       Deangelo Todd   1123.34
1        Nikolai Kent   2133.43
7       Shawn Winters   3489.25
3    Taliyah Martinez   8345.15
6  Jonathon Middleton  12764.50
9      Darryl Rosales  25645.39
```

### LIMIT

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
***Example Usage***

```py
>>>@bodo.jit
... def g1(df):
...    bc = bodosql.BodoSQLContext({"customers":df})
...    query = "SELECT name FROM customers LIMIT 4"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"customers":df})
...    query = "SELECT name FROM customers LIMIT 4 OFFSET 2"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "customerID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g1(customers_df) # LIMIT 4
               name
0     Deangelo Todd
1      Nikolai Kent
2        Eden Heath
3  Taliyah Martinez

>>>g2(customers_df) # LIMIT 4 OFFSET 2
               name
2        Eden Heath
3  Taliyah Martinez
4  Demetrius Chavez
5  Weston Jefferson
```

### NOT IN

The `#!sql IN` determines if a value can be chosen a list of options.
Currently, we support lists of literals or columns with matching
types:
```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <COLUMN_NAME> IN (<val1>, <val2>, ... <valN>)
```
For example:
```sql
SELECT A FROM table1 WHERE A IN (5, 10, 15, 20, 25)
```
***Example Usage***
```py
>>>@bodo.jit
... def g1(df):
...    bc = bodosql.BodoSQLContext({"payments":df})
...    query = "SELECT customerID FROM payments WHERE paymentType IN ('AMEX', 'WIRE')"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"payments":df})
...    query = "SELECT customerID FROM payments WHERE paymentType NOT IN ('AMEX', 'VISA')"
...    res = bc.sql(query)
...    return res

>>>payment_df = pd.DataFrame({
...     "customerID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE", "VISA", "VISA", "WIRE", "VISA", "AMEX"],
... })

>>>g1(payment_df) # IN
   customerID
2           2
4           4
7           7
9           9

>>>g2(payment_df) # NOT IN
   customerID
4           4
7           7
```

### NOT BETWEEN

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
***Example Usage***
```py
>>>@bodo.jit
... def g(df):
...    bc = bodosql.BodoSQLContext({"customers":df})
...    query = "SELECT name, balance FROM customers WHERE balance BETWEEN 1000 and 5000"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"customers":df})
...    query = "SELECT name, balance FROM customers WHERE balance NOT BETWEEN 100 and 10000"
...    res = bc.sql(query)
...    return res

>>>customers_df = pd.DataFrame({
...     "customerID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
...     "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez",
...                 "Demetrius Chavez","Weston Jefferson","Jonathon Middleton",
...                 "Shawn Winters","Keely Hutchinson", "Darryl Rosales",],
...     "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43, 68.34, 12764.50, 3489.25, 654.24, 25645.39]
... })

>>>g1(payment_df) # BETWEEN
            name  balance
0  Deangelo Todd  1123.34
1   Nikolai Kent  2133.43
7  Shawn Winters  3489.25

>>>g2(payment_df) # NOT BETWEEN
                 name   balance
2          Eden Heath     23.58
5    Weston Jefferson     68.34
6  Jonathon Middleton  12764.50
9      Darryl Rosales  25645.39
```

### CAST

THE `#!sql CAST` operator converts an input from one type to another. In
many cases casts are created implicitly, but this operator can be
used to force a type conversion.

The following casts are currently supported. Please refer to
`supported_dataframe_data_types` for
the Python types for each type keyword:

| From                             | To                              | Notes                                                                         |
|----------------------------------|---------------------------------|-------------------------------------------------------------------------------|
|`VARCHAR`                         |`VARCHAR`                        |                                                                               |
|`VARCHAR`                         |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`VARCHAR`                         |`FLOAT/DOUBLE`                   |                                                                               |
|`VARCHAR`                         |`DECIMAL`                        | Equivalent to `DOUBLE`. This may change in the future.                        |
|`VARCHAR`                         |`TIMESTAMP`                      |                                                                               |
|`VARCHAR`                         |`DATE`                           | Truncates to date but is still Timestamp type. This may change in the future. |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`VARCHAR`                        |                                                                               |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`FLOAT/DOUBLE`                   |                                                                               |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`DECIMAL`                        | Equivalent to `DOUBLE`. This may change in the future.                        |
|`TINYINT/SMALLINT/INTEGER/BIGINT` |`TIMESTAMP`                      |                                                                               |
|`FLOAT/DOUBLE`                    |`VARCHAR`                        |                                                                               |
|`FLOAT/DOUBLE`                    |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`FLOAT/DOUBLE`                    |`FLOAT/DOUBLE`                   |                                                                               |
|`FLOAT/DOUBLE`                    |`DECIMAL`                        | Equivalent to `DOUBLE`. This may change in the future                         |
|`TIMESTAMP`                       |`VARCHAR`                        |                                                                               |
|`TIMESTAMP`                       |`TINYINT/SMALLINT/INTEGER/BIGINT`|                                                                               |
|`TIMESTAMP`                       |`TIMESTAMP`                      |                                                                               |
|`TIMESTAMP`                       |`DATE`                           | Truncates to date but is still Timestamp type. This may change in the future. |

!!! note
    `#!sql CAST` correctness can often not be determined at compile time.
    Users are responsible for ensuring that conversion is possible
    (e.g. `#!sql CAST(str_col as INTEGER)`).

### ::

Infix cast operator. Equivalent to cast, but the format is `#!sql value::Typename`

### JOIN

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

***Example Usage***

```py
>>>@bodo.jit
... def g1(df1, df2):
...    bc = bodosql.BodoSQLContext({"customers":df1, "payments":df2})
...    query = "SELECT name, paymentType FROM customers JOIN payments ON customers.customerID = payments.customerID"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df1, df2):
...    bc = bodosql.BodoSQLContext({"customers":df1, "payments":df2})
...    query = "SELECT name, paymentType FROM customers FULL JOIN payments ON customers.customerID = payments.customerID"
...    res = bc.sql(query)
...    return res

>>>customer_df = pd.DataFrame({
...    "customerID": [0, 2, 4, 5, 7,],
...    "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez","Demetrius Chavez",],
...    "address": ["223 Iroquois LanenWest New York, NJ 07093","37 Depot StreetnTaunton, MA 02780",
...                "639 Maple St.nNorth Kingstown, RI 02852","93 Bowman Rd.nChester, PA 19013",
...                "513 Manchester Ave.nWindsor, CT 06095",],
...    "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43,]
... })
>>>payment_df = pd.DataFrame({
...     "customerID": [0, 1, 4, 6, 7],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE",],
... })

>>>g1(customer_df, payment_df) # INNER JOIN
               name paymentType
0     Deangelo Todd        VISA
1        Eden Heath        AMEX
2  Demetrius Chavez        WIRE

>>>g2(customer_df, payment_df) # OUTER JOIN
               name paymentType
0     Deangelo Todd        VISA
1      Nikolai Kent         NaN
2        Eden Heath        AMEX
3  Taliyah Martinez         NaN
4  Demetrius Chavez        WIRE
5               NaN        VISA
6               NaN        VISA
```

### NATURAL JOIN

A natural join is a type of join that provides an equality condition on all
columns with the same name and only returns 1 column for the keys. On cannot
be provided because it is implied but all join types can be provided.

```sql
SELECT <COLUMN_NAMES>
  FROM <LEFT_TABLE_NAME>
  NATURAL <JOIN_TYPE> <RIGHT_TABLE_NAME>
```
For example:
```sql
SELECT table1.A, table1.B FROM table1 NATURAL JOIN table2
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

***Example Usage***

```py
>>>@bodo.jit
... def g1(df1, df2):
...    bc = bodosql.BodoSQLContext({"customers":df1, "payments":df2})
...    query = "SELECT payments.* FROM customers NATURAL JOIN payments"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df1, df2):
...    bc = bodosql.BodoSQLContext({"customers":df1, "payments":df2})
...    query = "SELECT payments.* FROM customers NATURAL FULL JOIN payments"
...    res = bc.sql(query)
...    return res

>>>customer_df = pd.DataFrame({
...    "customerID": [0, 2, 4, 5, 7,],
...    "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez","Demetrius Chavez",],
...    "address": ["223 Iroquois LanenWest New York, NJ 07093","37 Depot StreetnTaunton, MA 02780",
...                "639 Maple St.nNorth Kingstown, RI 02852","93 Bowman Rd.nChester, PA 19013",
...                "513 Manchester Ave.nWindsor, CT 06095",],
...    "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43,]
... })
>>>payment_df = pd.DataFrame({
...     "customerID": [0, 1, 4, 6, 7],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE",],
... })

>>>g1(customer_df, payment_df) # INNER JOIN
   customerID paymentType
0           0        VISA
1           4        AMEX
2           7        WIRE

>>>g2(customer_df, payment_df) # OUTER JOIN
   customerID paymentType
0           0        VISA
1        <NA>        <NA>
2           4        AMEX
3        <NA>        <NA>
4           7        WIRE
5           1        VISA
6           6        VISA
```

### UNION

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

***Example Usage***

```py
>>>@bodo.jit
... def g1(df):
...    bc = bodosql.BodoSQLContext({"customers":df1, "payments":df2})
...    query = "SELECT name, paymentType FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE paymentType in ('WIRE')
...             UNION SELECT name, paymentType FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE balance < 1000"
...    res = bc.sql(query)
...    return res

>>>@bodo.jit
... def g2(df):
...    bc = bodosql.BodoSQLContext({"customers":df1, "payments":df2})
...    query = "SELECT name, paymentType FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE paymentType in ('WIRE')
...             UNION ALL SELECT name, paymentType FROM customers JOIN payments ON customers.customerID = payments.customerID WHERE balance < 1000"
...    res = bc.sql(query)
...    return res

>>>customer_df = pd.DataFrame({
...    "customerID": [0, 2, 4, 5, 7,],
...    "name": ["Deangelo Todd","Nikolai Kent","Eden Heath", "Taliyah Martinez","Demetrius Chavez",],
...    "address": ["223 Iroquois LanenWest New York, NJ 07093","37 Depot StreetnTaunton, MA 02780",
...                "639 Maple St.nNorth Kingstown, RI 02852","93 Bowman Rd.nChester, PA 19013",
...                "513 Manchester Ave.nWindsor, CT 06095",],
...    "balance": [1123.34, 2133.43, 23.58, 8345.15, 943.43,]
... })
>>>payment_df = pd.DataFrame({
...     "customerID": [0, 1, 4, 6, 7],
...     "paymentType": ["VISA", "VISA", "AMEX", "VISA", "WIRE",],
... })

>>>g1(customer_df, payment_df) # UNION
           name paymentType  balance
0  Demetrius Chavez        WIRE   943.43
0        Eden Heath        AMEX    23.58

>>>g2(customer_df, payment_df) # UNION ALL
            name paymentType  balance
0  Demetrius Chavez        WIRE   943.43
0        Eden Heath        AMEX    23.58
1  Demetrius Chavez        WIRE   943.43
```

### INTERSECT

The `#!sql INTERSECT` operator is used to calculate the intersection of
two `#!sql SELECT` statements:

```sql
SELECT <COLUMN_NAMES> FROM <TABLE1>
INTERSECT
SELECT <COLUMN_NAMES> FROM <TABLE2>
```

Each `#!sql SELECT` statement within the `#!sql INTERSECT` clause must have the
same number of columns. The columns must also have similar data
types. The output of the `#!sql INTERSECT` is the set of rows which are
present in both of the input SELECT statements. The `#!sql INTERSECT`
operator selects only the distinct values from the inputs.

### GROUP BY

The `#!sql GROUP BY` statement groups rows that have the same values
into summary rows, like "find the number of customers in each
country". The `#!sql GROUP BY` statement is often used with aggregate
functions to group the result-set by one or more columns:
```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <CONDITION>
GROUP BY <GROUP_EXPRESSION>
ORDER BY <COLUMN_NAMES>
```

For example:
```sql
SELECT MAX(A) FROM table1 GROUP BY B
```
`#!sql GROUP BY` statements also referring to columns by alias or
column number:
```sql
SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val
SELECT MAX(A), B FROM table1 GROUP BY 2
```

BodoSQL supports several subclauses that enable grouping by multiple different
sets of columns in the same `#!sql SELECT` statement. `#!sql GROUPING SETS` is the first. It is
equivalent to performing a group by for each specified set (setting each column not
present in the grouping set to null), and unioning the results. For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS (B, B, (B, C), ())
```

This is equivalent to:

```sql
SELECT * FROM
    (SELECT MAX(A), B, null FROM table1 GROUP BY B)
UNION
    (SELECT MAX(A), B, null FROM table1 GROUP BY B)
UNION
    (SELECT MAX(A), B, C FROM table1 GROUP BY B, C)
UNION
    (SELECT MAX(A), null, null FROM table1)
```

!!! note
    The above example is not valid BodoSQL code, as we do not support null literals.
    It is used only to show the null filling behavior.

`#!sql CUBE` is equivalent to grouping by all possible permutations of the specified set.
For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY CUBE(B, C)
```

Is equivalent to

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS ((B, C), (B), (C), ())
```

`#!sql ROLLUP` is equivalent to grouping by n + 1 grouping sets, where each set is constructed by dropping the rightmost element from the previous set, until no elements remain in the grouping set. For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY ROLLUP(B, C, D)
```

Is equivalent to

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS ((B, C, D), (B, C), (B), ())
```

`#!sql CUBE` and `#!sql ROLLUP` can be nested into a `#!sql GROUPING SETS` clause. For example:

```sql
SELECT MAX(A), B, C GROUP BY GROUPING SETS (ROLLUP(B, C, D), CUBE(B, C), (A))
```

Which is equivalent to

```sql
SELECT MAX(A), B, C GROUP BY GROUPING SETS ((B, C, D), (B, C), (B), (), (B, C), (B), (C), (), (A))
```

### HAVING

The `#!sql HAVING` clause is used for filtering with `#!sql GROUP BY`.
`#!sql HAVING` applies the filter after generating the groups, whereas
`#!sql WHERE` applies the filter before generating any groups:

```sql
SELECT column_name(s)
FROM table_name
WHERE condition
GROUP BY column_name(s)
HAVING condition
```

For example:
```sql
SELECT MAX(A) FROM table1 GROUP BY B HAVING C < 0
```
`#!sql HAVING` statements also referring to columns by aliases used in
the `#!sql GROUP BY`:
```sql
SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val HAVING val 5
```

### QUALIFY

`#!sql QUALIFY` is similar to `#!sql HAVING`, except it applies filters after computing the results of at least one window function. `#!sql QUALIFY` is used after using `#!sql WHERE` and `#!sql HAVING`.

For example:

```sql
SELECT column_name(s),
FROM table_name
WHERE condition
GROUP BY column_name(s)
HAVING condition
QUALIFY MAX(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 1 FOLLOWING AND 1 PRECEDING) > 1
```

Is equivalent to

```sql
SELECT column_name(s) FROM
    (SELECT column_name(s), MAX(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 1 FOLLOWING AND 1 PRECEDING) as window_output
    FROM table_name
    WHERE condition
    GROUP BY column_name(s)
    HAVING condition)
WHERE window_output > 1
```

### CASE

The `#!sql CASE` statement goes through conditions and returns a value
when the first condition is met:
```sql
SELECT CASE WHEN cond1 THEN value1 WHEN cond2 THEN value2 ... ELSE valueN END
```
For example:
```sql
SELECT (CASE WHEN A 1 THEN A ELSE B END) as mycol FROM table1
```
If the types of the possible return values are different, BodoSQL
will attempt to cast them all to a common type, which is currently
undefined behavior. The last else clause can optionally be
excluded, in which case, the `#!sql CASE` statement will return null if
none of the conditions are met. For example:
```sql
SELECT (CASE WHEN A < 0 THEN 0 END) as mycol FROM table1
```
is equivalent to:
```sql
SELECT (CASE WHEN A < 0 THEN 0 ELSE NULL END) as mycol FROM table1
```

### LIKE

The `#!sql LIKE` clause is used to filter the strings in a column to
those that match a pattern:
```sql
SELECT column_name(s) FROM table_name WHERE column LIKE pattern
```
In the pattern we support the wildcards `#!sql %` and `#!sql _`. For example:
```sql
SELECT A FROM table1 WHERE B LIKE '%py'
```

### GREATEST

The `#!sql GREATEST` clause is used to return the largest value from a
list of columns:
```sql
SELECT GREATEST(col1, col2, ..., colN) FROM table_name
```
For example:
```sql
SELECT GREATEST(A, B, C) FROM table1
```

### LEAST

The `#!sql LEAST` clause is used to return the smallest value from a
list of columns:
```sql
SELECT LEAST(col1, col2, ..., colN) FROM table_name
```
For example:
```sql
SELECT LEAST(A, B, C) FROM table1
```

### PIVOT

The `#!sql PIVOT` clause is used to transpose specific data rows in one
or more columns into a set of columns in a new DataFrame:
```sql
SELECT col1, ..., colN FROM table_name PIVOT (
    AGG_FUNC_1(colName or pivotVar) AS alias1, ...,  AGG_FUNC_N(colName or pivotVar) as aliasN
    FOR pivotVar IN (ROW_VALUE_1 as row_alias_1, ..., ROW_VALUE_N as row_alias_N)
)
```
`#!sql PIVOT` produces a new column for each pair of pivotVar and
aggregation functions.

For example:
```sql
SELECT single_sum_a, single_avg_c, triple_sum_a, triple_avg_c FROM table1 PIVOT (
    SUM(A) AS sum_a, AVG(C) AS avg_c
    FOR A IN (1 as single, 3 as triple)
)
```
Here `#!sql single_sum_a` will contain sum(A) where `#!sql A = 1`,
single_avg_c will contain AVG(C) where `#!sql A = 1` etc.

If you explicitly specify other columns as the output, those
columns will be used to group the pivot columns. For example:
```sql
SELECT B, single_sum_a, single_avg_c, triple_sum_a, triple_avg_c FROM table1 PIVOT (
    SUM(A) AS sum_a, AVG(C) AS avg_c
    FOR A IN (1 as single, 3 as triple)
)
```
Contains 1 row for each unique group in B. The pivotVar can also
require values to match in multiple columns. For example:
```sql
SELECT * FROM table1 PIVOT (
    SUM(A) AS sum_a, AVG(C) AS avg_c
    FOR (A, B) IN ((1, 4) as col1, (2, 5) as col2)
)
```

### WITH

The `#!sql WITH` clause can be used to name sub-queries:
```sql
WITH sub_table AS (SELECT column_name(s) FROM table_name)
SELECT column_name(s) FROM sub_table
```
For example:
```sql
WITH subtable as (SELECT MAX(A) as max_al FROM table1 GROUP BY B)
SELECT MAX(max_val) FROM subtable
```

### Aliasing

SQL aliases are used to give a table, or a column in a table, a
temporary name:

```sql
SELECT <COLUMN_NAME> AS <ALIAS>
FROM <TABLE_NAME>
```

For example:
```sql
Select SUM(A) as total FROM table1
```

We strongly recommend using aliases for the final outputs of any
queries to ensure all column names are predictable.

### Operators

#### Arithmetic
-   BodoSQL currently supports the following arithmetic
    operators:

    -   `+` (addition)
    -   `-` (subtraction)
    -   `*` (multiplication)
    -   `/` (true division)
    -   `%` (modulo)

#### Comparison
-   BodoSQL currently supports the following comparison
    operators:

    -   `=` (equal to)
    -   `>` (greater than)
    -   `<` (less than)
    -   `>=` (greater than or equal to)
    -   `<=` (less than or equal to)
    -   `<>` (not equal to)
    -   `!=` (not equal to)
    -   `<=>` (equal to or both inputs are null)

#### Logical
-   BodoSQL currently supports the following logical operators:

    -   `#!sql AND`
    -   `#!sql OR`
    -   `#!sql NOT`

#### String
-   BodoSQL currently supports the following string operators:

    -   `||` (string concatenation)

###  Numeric Functions

Except where otherwise specified, the inputs to each of these
functions can be any numeric type, column or scalar. Here is an
example using MOD:

```sql
SELECT MOD(12.2, A) FROM table1
```

BodoSQL Currently supports the following Numeric Functions:

#### ABS
-   `#!sql ABS(n)`

    Returns the absolute value of n

#### COS
-   `#!sql COS(n)`

    Calculates the Cosine of n

#### SIN
-   `#!sql SIN(n)`

    Calculates the Sine of n

#### TAN
-   `#!sql TAN(n)`

    Calculates the Tangent of n


#### COTAN
-   `#!sql COTAN(X)`

    Calculates the Cotangent of `X`


#### ACOS
-   `#!sql ACOS(n)`

    Calculates the Arccosine of n

#### ASIN
-   `#!sql AIN(n)`

    Calculates the Arcsine of n

#### ATAN
-   `#!sql ATAN(n)`

    Calculates the Arctangent of n

#### ATAN2
-   `#!sql ATAN2(A, B)`

    Calculates the Arctangent of `A` divided by `B`


#### CEIL
-   `#!sql CEIL(X[, scale])`

    Converts X to the specified scale, rounding towards positive
    infinity. For example, `scale=0` rounds up to the nearest integer,
    `scale=2` rounds up to the nearest `0.01`, and `scale=-1` rounds
    up to the nearest multiple of 10.

#### CEILING
-   `#!sql CEILING(X)`

    Equivalent to `#!sql CEIL`

#### FLOOR
-   `#!sql FLOOR(X[, scale])`

    Converts X to the specified scale, rounding towards negative
    infinity. For example, `scale=0` down up to the nearest integer,
    `scale=2` rounds down to the nearest `0.01`, and `scale=-1` rounds
    down to the nearest multiple of 10.

#### DEGREES
-   `#!sql DEGREES(X)`

    Converts a value in radians to the corresponding value in
    degrees

#### RADIANS
-   `#!sql RADIANS(X)`

    Converts a value in radians to the corresponding value in
    degrees

#### LOG10
-   `#!sql LOG10(X)`

    Computes Log base 10 of x. Returns NaN for negative inputs,
    and -inf for 0 inputs.

#### LOG
-   `#!sql LOG(X)`

    Equivalent to `#!sql LOG10(x)`

#### LOG10
-   `#!sql LOG10(X)`

    Computes Log base 2 of x. Returns `NaN` for negative inputs,
    and `#!sql -inf` for 0 inputs.

#### LN
-   `#!sql LN(X)`

    Computes the natural log of x. Returns `NaN` for negative
    inputs, and `#!sql -inf` for 0 inputs.

#### MOD
-   `#!sql MOD(A,B)`

    Computes A modulo B (behavior analogous to the C library function `fmod`). Returns `NaN` if B is 0 or if A is inf.

#### CONV
-   `#!sql CONV(X, current_base, new_base)`

    `#!sql CONV` takes a string representation of an integer value,
    it's current_base, and the base to convert that argument
    to. `#!sql CONV` returns a new string, that represents the value in
    the new base. `#!sql CONV` is only supported for converting to/from
    base 2, 8, 10, and 16.

    For example:

    ```sql
    CONV('10', 10, 2) =='1010'
    CONV('10', 2, 10) =='2'
    CONV('FA', 16, 10) =='250'
    ```
#### SQRT
-   `#!sql SQRT(X)`

    Computes the square root of x. Returns `NaN` for negative
    inputs, and `#!sql -inf` for 0 inputs.

#### PI
-   `#!sql PI()`

    Returns the value of `#!sql PI`

#### POW, POWER
-   `#!sql POW(A, B), POWER(A, B)`

    Returns A to the power of B. Returns `NaN` if A is negative,
    and B is a float. `#!sql POW(0,0)` is 1

#### EXP
-   `#!sql EXP(X)`

    Returns e to the power of X

#### SIGN
-   `#!sql SIGN(X)`

    Returns 1 if X > 0, -1 if X < 0, and 0 if X = 0

#### ROUND
-   `#!sql ROUND(X[, num_decimal_places])`

    Rounds X to the specified number of decimal places

#### TRUNCATE
-   `#!sql TRUNCATE(X[, num_decimal_places])`

    Equivalent to `#!sql ROUND(X, num_decimal_places)`. If `num_decimal_places`
    is not supplied, it defaults to 0.


#### TRUNC
-   `#!sql TRUNC(X[, num_decimal_places])`

    Equivalent to `#!sql TRUNC(X[, num_decimal_places])` if `X` is numeric.
    Note that `TRUNC` is overloaded and may invoke the timestamp function
    `TRUNC` if `X` is a date or time expression.


#### BITAND
-   `#!sql BITAND(A, B)`

    Returns the bitwise-and of its inputs.


#### BITOR
-   `#!sql BITOR(A, B)`

    Returns the bitwise-or of its inputs.


#### BITXOR
-   `#!sql BITOR(A, B)`

    Returns the bitwise-xor of its inputs.


#### BITNOT
-   `#!sql BITNOT(A)`

    Returns the bitwise-negation of its input.



#### BITSHIFTLEFT
-   `#!sql BITSHIFTLEFT(A, B)`

    Returns the bitwise-leftshift of its inputs.

    !!! note
        - The output is always of type int64.
        - Undefined behavior when B is negative or too large.


#### BITSHIFTRIGHT
-   `#!sql BITSHIFTRIGHT(A, B)`

    Returns the bitwise-rightshift of its inputs.
    Undefined behavior when B is negative or
    too large.


#### GETBIT
-   `#!sql GETBIT(A, B)`

    Returns the bit of A corresponding to location B,
    where 0 is the rightmost bit. Undefined behavior when
    B is negative or too large.


#### BOOLAND
-   `#!sql BOOLAND(A, B)`

    Returns true when `A` and `B` are both non-null non-zero.
    Returns false when one of the arguments is zero and the
    other is either zero or `NULL`. Returns `NULL` otherwise.


#### BOOLOR
-   `#!sql BOOLOR(A, B)`

    Returns true if either `A` or `B` is non-null and non-zero.
    Returns false if both `A` and `B` are zero. Returns `NULL` otherwise.


#### BOOLXOR
-   `#!sql BOOLXOR(A, B)`

    Returns true if one of `A` and `B` is zero and the other is non-zero.
    Returns false if `A` and `B` are both zero or both non-zero. Returns
    `NULL` if either `A` or `B` is `NULL`.


#### BOOLNOT
-   `#!sql BOOLNOT(A)`

    Returns true if `A` is zero. Returns false if `A` is non-zero. Returns
    `NULL` if `A` is `NULL`.


#### REGR_VALX
-   `#!sql REGR_VALX(Y, X)`

    Returns `NULL` if either input is `NULL`, otherwise `X`


#### REGR_VALY
-   `#!sql REGR_VALY(Y, X)`

    Returns `NULL` if either input is `NULL`, otherwise `Y`


#### HASH
-   `#!sql HASH(A, B, C, ...)`

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


###  Data Generation Functions

BodoSQL Currently supports the following data generaiton functions:

#### RANDOM
-   `#!sql RANDOM()`

    Outputs a random 64-bit integer. If used inside of a select statement with
    a table, the number of random values will match the number of rows in the
    input table (and each value should be randomly and independently generated).
    Note that running with multiple processors may affect the randomization
    results.

    !!! note
        Currently, BodoSQL does not support the format of `#!sql RANDOM()` that
        takes in a seed value.

    !!! note
        At present, aliases to `RANDOM` calls occasionally produce unexpected
        behavior. For certain SQL operations, calling `RANDOM` and storing the
        result with an alias, then later re-using that alias may result in
        another call to `RANDOM`. This behavior is somewhat rare.


#### UNIFORM
-   `#!sql UNIFORM(lo, hi, gen)`

    Outputs a random number uniformly distributed in the interval `[lo, hi]`.
    If `lo` and `hi` are both integers, then the output is an integer between
    `lo` and `hi` (including both endpoints). If either `lo` or `hi` is a float,
    the output is a random float between them. The values of `gen` are used to
    seed the randomness, so if `gen` is all distinct values (or is randomly
    generated) then the output of `UNIFORM` should be random. However, if 2
    rows have the same `gen` value they will produce the same output value.


### Aggregation Functions

BodoSQL Currently supports the following Aggregation Functions on
all types:


#### COUNT
-   `#!sql COUNT`

    Count the number of elements in a column or group.


#### ANY_VALUE
-   `#!sql ANY_VALUE`

    Select an arbitrary value.

    !!! note
        Currently, BodoSQL always selects the first value, but this is subject to change at any time.


In addition, BodoSQL also supports the following functions on
numeric types


#### AVG
-   `#!sql AVG`

    Compute the mean for a column.

#### MAX
-   `#!sql MAX`

    Compute the max value for a column.

#### MIN
-   `#!sql MIN`

    Compute the min value for a column.

#### STDDEV
-   `#!sql STDDEV`

    Compute the standard deviation for a column with N - 1
    degrees of freedom.

#### STDDEV_SAMP
-   `#!sql STDDEV_SAMP`

    Compute the standard deviation for a column with N - 1
    degrees of freedom.

#### STDDEV_POP
-   `#!sql STDDEV_POP`

    Compute the standard deviation for a column with N degrees
    of freedom.

#### SUM
-   `#!sql SUM`

    Compute the sum for a column.

#### COUNT_IF
-   `#!sql COUNT_IF`

    Compute the total number of occurrences of `#!sql true` in a column
    of booleans. For example:

    ```sql
    SELECT COUNT_IF(A) FROM table1
    ```

    Is equivalent to
    ```sql
    SELECT SUM(CASE WHEN A THEN 1 ELSE 0 END) FROM table1
    `#!sql ``


#### LISTAGG
-   `LISTAGG(str_col[, delimeter]) [WITHIN GROUP (ORDER BY order_col)]`

    Concatenates all of the strings in `str_col` within each group into a single
    string seperated by the characters in the string `delimiter`. If no delimiter
    is provided, an empty string is used by default.

    Optionally allows using a `WITHIN GROUP` clause to specify how the strings should
    be ordered before being concatenated. If no clause is specified, then the ordering
    is unpredictable.


#### MODE
-   `#!sql MODE`

    Returns the most frequent element in a group, or `NULL` if the group is empty.

    !!! note
        This aggregation function is currently only supported with a `GROUP BY` clause.

#### ARRAY_AGG
-   `#!sql ARRAY_AGG([DISTINCT] A) [WITHIN GROUP(ORDER BY orderby_terms)]`

    Combines all the values in column `A` within each group into a single array.

    Optionally allows using a `WITHIN GROUP` clause to specify how the values should
    be ordered before being combined into an array. If no clause is specified, then the ordering
    is unpredictable. Nulls will not be included in the arrays.

    If the `DISTINCT` keyword is provided, then duplicate elements are removed from each of
    the arrays. However, if this keyword is provied and a `WITHIN GROUP` clause is also provided,
    then the `WITHIN GROUP` clause can only refer to the same column as the aggregation input.

    !!! note
        This aggregation function is currently only supported with a `GROUP BY` clause,
        and on numerical data (integers, floats, etc.) or string/binary data.


#### OBJECT_AGG
-   `#!sql OBJECT_AGG(K, V)`

    Combines the data from columns `K` and `V` into a JSON object where the rows of
    column `K` are the field names and the rows of column `V` are the values.

    !!! note
        This aggregation function is currently only supported with a `GROUP BY` clause.


#### APPROX_PERCENTILE
-   `#!sql APPROX_PERCENTILE(A, q)`

    Returns the approximate value of the `q`-th percentile of column `A` (e.g.
    0.5 = median, or 0.9 = the 90th percentile). `A` can be any numeric column,
    and `q` can be any scalar float between zero and one.

    The approximation is calculated using the t-digest algorithm.

#### PERCENTILE_CONT
-   `#!sql APPROX_PEPERCENTILE_CONTRCENTILE(q) WITHIN GROUP (ORDER BY A)`

    Computes the exact value of the `q`-th percentile of column `A` (e.g.
    0.5 = median, or 0.9 = the 90th percentile). `A` can be any numeric column,
    and `q` can be any scalar float between zero and one.

    If no value lies exactly at the desired percentile, the two nearest
    values are linearly interpolated. For example, consider the dataset `[2, 8, 25, 40]`.
    If we sought the percentile `q=0.25` we would be looking for the value
    at index 0.75. There is no value at index 0.75, so we linearly interpolate
    between 2 and 8 to get 6.5.

#### PERCENTILE_DISC
-   `#!sql PERCENTILE_DISC(q) WITHIN GROUP (ORDER BY A)`

    Computes the exact value of the `q`-th percentile of column `A` (e.g.
    0.5 = median, or 0.9 = the 90th percentile). `A` can be any numeric column,
    and `q` can be any scalar float between zero and one.

    This function differs from `PERCENTILE_CONT` in that it always outputs a
    value from the original array. The value it chooses is the smallest value
    in `A` such that the `CUME_DIST` of all values in the column `A` is greater
    than or equal to `q`. For example, consider the dataset `[2, 8, 8, 40]`.
    The `CUME_DIST` of each of these values is `[0.25, 0.75, 0.75, 1.0]`.
    If we sought the percentile `q=0.6` we would output 8 since it has the
    smallest `CUME_DIST` that is `>=0.6`.

#### VARIANCE
-   `#!sql VARIANCE`

    Compute the variance for a column with N - 1 degrees of
    freedom.

#### VAR_SAMP
-   `#!sql VAR_SAMP`

    Compute the variance for a column with N - 1 degrees of
    freedom.

#### VAR_POP
-   `#!sql VAR_POP`

    Compute the variance for a column with N degrees of freedom.

#### SKEW
-   `#!sql SKEW`

    Compute the skew of a column

#### KURTOSIS
-   `#!sql KURTOSIS`

    Compute the kurtosis of a column

#### BITOR_AGG
-   `#!sql BITOR_AGG`

    Compute the bitwise OR of every input
    in a group, returning `#!sql NULL` if there are no non-`#!sql NULL` entries.
    Accepts floating point values, integer values, and strings. Strings are interpreted
    directly as numbers, converting to 64-bit floating point numbers.


#### BOOLOR_AGG
-   `#!sql BOOLOR_AGG`

    Compute the logical OR of the boolean value of every input
    in a group, returning `#!sql NULL` if there are no non-`#!sql NULL` entries, otherwise
    returning True if there is at least 1 non-zero entry. This is supported for
    numeric and boolean types.

#### BOOLAND_AGG
-   `#!sql BOOLAND_AGG`

    Compute the logical AND of the boolean value of every input
    in a group, returning `#!sql NULL` if there are no non-`#!sql NULL` entries, otherwise
    returning True if all non-`#!sql NULL` entries are also non-zero. This is supported for
    numeric and boolean types.

#### BOOLXOR_AGG
-   `#!sql BOOLXOR_AGG`

    Returns `#!sql NULL` if there are no non-`#!sql NULL` entries, otherwise
    returning True if exactly one non-`#!sql NULL` entry is also non-zero (this is
    counterintuitive to how the logical XOR is normally thought of). This is
    supported for numeric and boolean types.

All aggregate functions have the syntax:

```sql
SELECT AGGREGATE_FUNCTION(<COLUMN_EXPRESSION>)
FROM <TABLE_NAME>
GROUP BY <COLUMN_NAMES>
```

These functions can be used either in a groupby clause, where they
will be computed for each group, or by itself on an entire column
expression. For example:

```sql
SELECT AVG(A) FROM table1 GROUP BY B

SELECT COUNT(Distinct A) FROM table1
```

### Timestamp Functions

BodoSQL currently supports the following Timestamp functions:

#### DATEDIFF
-   `#!sql DATEDIFF(timestamp_val1, timestamp_val2)`

    Computes the difference in days between two Timestamp
    values (timestamp_val1 - timestamp_val2)


-   `#!sql DATEDIFF(unit, timestamp_val1, timestamp_val2)`

    Computes the difference between two Timestamp
    values (timestamp_val2 - timestamp_val1) in terms of unit

    Allows the following units, with the specified
    abbreviations as string literals:

    -   YEAR: `year`, `years`, `yr`, `yrs`, `y`, `yy`, `yyy`, `yyyy`
    -   QUARTER: `quarter`, `quarters`, `q`, `qtr`, `qtrs`
    -   MONTH: `month`, `months`, `mm`, `mon`, `mons`
    -   WEEK: `week`, `weeks`, `weekofyear`, `w`, `wk`, `woy`, `wy`
    -   DAY: `day`, `days`, `dayofmonth`, `d`, `dd`
    -   HOUR: `hour`, `hours`, `hrs`, `h`, `hr`, `hrs`
    -   MINUTE: `minute`, `minutes`, `m`, `mi`, `min`, `mins`
    -   SECOND: `second`, `seconds`, `s`, `sec`, `secs`
    -   MILLISECOND: `millisecond`, `milliseconds`, `ms`, `msecs`
    -   MICROSECOND: `microsecond`, `microseconds`, `us`, `usec`
    -   NANOSECOND: `nanosecond`, `nanoseconds`, `nanosec`, `nsec`, `nsecs`, `nsecond`, `ns`, `nanonsecs`

#### STR_TO_DATE
-   `#!sql STR_TO_DATE(str_val, literal_format_string)`

    Converts a string value to a Timestamp value given a
    literal format string. If a year, month, and day value is
    not specified, they default to 1900, 01, and 01
    respectively. Will throw a runtime error if the string
    cannot be parsed into the expected values. See [`DATE_FORMAT`][date_format]
    for recognized formatting characters.

    For example:

    ```sql
    STR_TO_DATE('2020 01 12', '%Y %m %d') ==Timestamp '2020-01-12'
    STR_TO_DATE('01 12', '%m %d') ==Timestamp '1900-01-12'
    STR_TO_DATE('hello world', '%Y %m %d') ==RUNTIME ERROR
    ```

#### DATE_FORMAT
-   `#!sql DATE_FORMAT(timestamp_val, literal_format_string)`

    Converts a timestamp value to a String value given a
    scalar format string.

    Recognized formatting characters:

    -   `#!sql %i` Minutes, zero padded (00 to 59)
    -   `#!sql %M` Full month name (January to December)
    -   `#!sql %r` Time in format in the format (hh\:mm\:ss AM/PM)
    -   `#!sql %s` Seconds, zero padded (00 to 59)
    -   `#!sql %T` Time in format in the format (hh\:mm\:ss)
    -   `#!sql %T` Time in format in the format (hh\:mm\:ss)
    -   `#!sql %u` week of year, where monday is the first day of the week(00 to 53)
    -   `#!sql %a` Abbreviated weekday name (sun-sat)
    -   `#!sql %b` Abbreviated month name (jan-dec)
    -   `#!sql %f` Microseconds, left padded with 0's, (000000 to 999999)
    -   `#!sql %H` Hour, zero padded (00 to 23)
    -   `#!sql %j` Day Of Year, left padded with 0's (001 to 366)
    -   `#!sql %m` Month number (00 to 12)
    -   `#!sql %p` AM or PM, depending on the time of day
    -   `#!sql %d` Day of month, zero padded (01 to 31)
    -   `#!sql %Y` Year as a 4 digit value
    -   `#!sql %y` Year as a 2 digit value, zero padded (00 to 99)
    -   `#!sql %U` Week of year, where Sunday is the first day of the week
        (00 to 53)
    -   `#!sql %S` Seconds, zero padded (00 to 59)

    For example:

    ```sql
    DATE_FORMAT(Timestamp '2020-01-12', '%Y %m %d') =='2020 01 12'
    DATE_FORMAT(Timestamp '2020-01-12 13:39:12', 'The time was %T %p. It was a %u') =='The time was 13:39:12 PM. It was a Sunday'
    ```


#### DATE_FROM_PARTS
-   `DATE_FROM_PARTS(year, month, day)`

    Constructs a date from the integer inputs specified, e.g. `(2020, 7, 4)`
    will output July 4th, 2020.

    Note: month does not have to be in the 1-12 range, and day does not have to
    be in the 1-31 range. Values out of bounds are overflowed logically,
    e.g. `(2020, 14, -1)` will output January 31st, 2021.


#### DATEFROMPARTS
-   `DATEFROMPARTS(year, month, day)`

    Equivalent to `DATE_FROM_PARTS`


#### TIME_FROM_PARTS
-   `#!sql TIME_FROM_PARTS(integer_hour_val, integer_minute_val, integer_second_val [, integer_nanoseconds_val])`

    Creates a time from individual numeric components. Usually,
    `integer_hour_val` is in the 0-23 range, `integer_minute_val` is in the 0-59
    range, `integer_second_val` is in the 0-59 range, and
    `integer_nanoseconds_val` (if provided) is a 9-digit integer.
    ```sql
    TIMEFROMPARTS(12, 34, 56, 987654321)
    12:34:56.987654321
    ```


#### TIMEFROMPARTS
-   `#!sql TIMEFROMPARTS(integer_hour_val, integer_minute_val, integer_second_val [, integer_nanoseconds_val])`

    See TIME_FROM_PARTS.

    ```sql
    TIMEFROMPARTS(12, 34, 56, 987654321)
    12:34:56.987654321
    ```

#### TIMESTAMP_FROM_PARTS
-   `#!sql TIMESTAMP_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`
-   `#!sql TIMESTAMP_FROM_PARTS(date_expr, time_expr)`
    The first overload is equivalent to `DATE_FROM_PARTS` but also takes in an
    hour, minute and second (which can be out of bounds just like the
    month/day). Optionally takes in a nanosecond value, and a timezone value
    for the output. If the timezone is not specified, the output is
    timezone-naive. Note that if any numeric argument cannot be converted to
    an int64, then it will become NULL.

    Note: timezone argument is not supported at this time.

    The second overload constructs the timestamp by combining the date and time
    arguments. The output of this function is always timestamp-naive.


#### TIMESTAMPFROMPARTS
-   `TIMESTAMPFROMPARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`
-   `TIMESTAMPFROMPARTS(date_expr, time_expr)`

    Equivalent to `TIMESTAMP_FROM_PARTS`


#### TIMESTAMP_NTZ_FROM_PARTS
-   `TIMESTAMP_NTZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`
-   `TIMESTAMP_NTZ_FROM_PARTS(date_expr, time_expr)`

    Equivalent to `TIMESTAMP_FROM_PARTS` but without the optional timezone
    argument in the first overload. The output is always timezone-naive.


#### TIMESTAMPNTZFROMPARTS
-   `TIMESTAMP_NTZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`
-   `TIMESTAMP_NTZ_FROM_PARTS(date_expr, time_expr)`

    Equivalent to `TIMESTAMP_NTZ_FROM_PARTS`


#### TIMESTAMP_LTZ_FROM_PARTS
-   `TIMESTAMP_LTZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`

    Equivalent to `TIMESTAMP_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`
    but without the optional timezone argument in the first overload. The output
    is always timezone-aware using the local timezone.


#### TIMESTAMPLTZFROMPARTS
-   `TIMESTAMP_LTZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond])`

    Equivalent to `TIMESTAMP_LTZ_FROM_PARTS`


#### TIMESTAMP_TZ_FROM_PARTS
-   `TIMESTAMP_TZ_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`

    Equivalent to `TIMESTAMP_FROM_PARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`
    except the default behavior if no timezone is provided is to use the local
    timezone instead of timezone-naive.

    Note: timezone argument is not supported at this time.


#### TIMESTAMPTZFROMPARTS
-   `TIMESTAMPTZFROMPARTS(year, month, day, hour, minute, second[, nanosecond[, timezone]])`

    Equivalent to `TIMESTAMP_TZ_FROM_PARTS`


#### DATEADD
-   `#!sql DATEADD(unit, amount, timestamp_val)`

    Computes a timestamp column by adding the amount of the specified unit
    to the timestamp val. For example, `#!sql DATEADD('day', 3, T)` adds 3 days to
    column `T`. Allows the following units, with the specified
    abbreviations as string literals:

    -   YEAR: `year`, `years`, `yr`, `yrs`, `y`, `yy`, `yyy`, `yyyy`
    -   QUARTER: `quarter`, `quarters`, `q`, `qtr`, `qtrs`
    -   MONTH: `month`, `months`, `mm`, `mon`, `mons`
    -   WEEK: `week`, `weeks`, `weekofyear`, `w`, `wk`, `woy`, `wy`
    -   DAY: `day`, `days`, `dayofmonth`, `d`, `dd`
    -   HOUR: `hour`, `hours`, `hrs`, `h`, `hr`, `hrs`
    -   MINUTE: `minute`, `minutes`, `m`, `mi`, `min`, `mins`
    -   SECOND: `second`, `seconds`, `s`, `sec`, `secs`
    -   MILLISECOND: `millisecond`, `milliseconds`, `ms`, `msecs`
    -   MICROSECOND: `microsecond`, `microseconds`, `us`, `usec`
    -   NANOSECOND: `nanosecond`, `nanoseconds`, `nanosec`, `nsec`, `nsecs`, `nsecond`, `ns`, `nanonsecs`

    Supported with timezone-aware data.

-   `#!sql DATEADD(timestamp_val, amount)`

    Equivalent to `#!sql DATEADD('day', amount, timestamp_val)`


#### TIMEADD
-   `#!sql TIMEADD(unit, amount, timestamp_val)`

    Equivalent to `#!sql DATEADD`.


#### TIMESTAMPADD
-   `#!sql TIMESTAMPADD(unit, amount, timestamp_val)`

    Equivalent to `#!sql DATEADD`.


#### DATE_ADD
-   `#!sql DATE_ADD(timestamp_val, interval)`

    Computes a timestamp column by adding an interval column/scalar to a
    timestamp value. If the first argument is a string representation of a
    timestamp, Bodo will cast the value to a timestamp.


-   `#!sql DATE_ADD(timestamp_val, amount)`

    Equivalent to `#!sql DATE_ADD('day', amount, timestamp_val)`


#### DATE_SUB
-   `#!sql DATE_SUB(timestamp_val, interval)`

    Computes a timestamp column by subtracting an interval column/scalar
    to a timestamp value. If the first argument is a string representation
    of a timestamp, Bodo will cast the value to a timestamp.

#### DATE_TRUNC
-   `#!sql DATE_TRUNC(str_literal, timestamp_val)`

    Truncates a timestamp to the provided str_literal field.
    str_literal must be a compile time constant and one of:

    -   "MONTH"
    -   "WEEK"
    -   "DAY"
    -   "HOUR"
    -   "MINUTE"
    -   "SECOND"
    -   "MILLISECOND"
    -   "MICROSECOND"
    -   "NANOSECOND"

#### TRUNC
-   `#!sql TRUNC(timestamp_val, str_literal)`

    Equivalent to `#!sql DATE_TRUNC(str_literal, timestamp_val)`. The
    argument order is reversed when compared to `DATE_TRUNC`. Note that `TRUNC`
    is overloaded, and may invoke the numeric function `TRUNCATE` if the
    arguments are numeric.

#### TIME_SLICE
-   `#!sql TIME_SLICE(date_or_time_expr, slice_length, unit[, start_or_end])`

    Calculates one of the endpoints of a "slice" of time containing the date
    specified by `date_or_time_expr` where each slice has length of time corresponding
    to `slice_length` times the date/time unit specified by `unit`. The slice
    start/ends are always aligned to the unix epoch `1970-01-1` (at midnight). The fourth argument
    specifies whether to return the begining or the end of the slice
    (`'START'` for begining, `'END'` for end), where the default is `'START'`.

    For example, `#!sql TIME_SLICE(T, 3, 'YEAR')` would return the timestamp
    corresponding to the begining of the first 3-year window (aligned with
    1970) that contains timestamp `T`. So `T = 1995-7-4 12:30:00` would
    output `1994-1-1` for `'START'` or `1997-1-1` for `'END'`.


#### NOW
-   `#!sql NOW()`

    Computes a timestamp equal to the current time in the session's timezone.
    By default, the current timezone is UTC, and it can be updated as a parameter
    when using the Snowflake Catalog.


#### LOCALTIMESTAMP
-   `#!sql LOCALTIMESTAMP()`

    Equivalent to `#!sql NOW`


#### CURRENT_TIMESTAMP
-   `#!sql CURRENT_TIMESTAMP()`

    Equivalent to `#!sql NOW`

#### GETDATE
-   `#!sql GETDATE()`

    Equivalent to `#!sql NOW`

#### SYSTIMESTAMP
-   `#!sql SYSTIMESTAMP()`

    Equivalent to `#!sql NOW`

#### LOCALTIME
-   `#!sql LOCALTIME()`

    Computes a time equal to the current time in the session's timezone.
    By default the current time is in local time, and it can be updated as a
    parameter when using the Snowflake Catalog.

#### CURRENT_TIME
-   `#!sql CURRENT_TIME()`

    Equivalent to `#!sql LOCALTIME`

#### CURDATE
-   `#!sql CURDATE()`

    Computes a timestamp equal to the current system time, excluding the
    time information

#### CURRENT_DATE
-   `#!sql CURRENT_DATE()`

    Equivalent to `#!sql CURDATE`

#### EXTRACT
-   `#!sql EXTRACT(TimeUnit from timestamp_val)`

    Extracts the specified TimeUnit from the supplied date.

    Allowed TimeUnits are:

    -   `MICROSECOND`
    -   `MINUTE`
    -   `HOUR`
    -   `DAY` (Day of Month)
    -   `DOY` (Day of Year)
    -   `DOW` (Day of week)
    -   `WEEK`
    -   `MONTH`
    -   `QUARTER`
    -   `YEAR`

    TimeUnits are not case-sensitive.


#### DATE_PART
-   `#!sql DATE_PART(unit, timestamp_val)`

    Equivalent to `#!sql EXTRACT(unit FROM timestamp_val)` with the following unit
    string literals:

    -   YEAR: `year`, `years`, `yr`, `yrs`, `y`, `yy`, `yyy`, `yyyy`
    -   QUARTER: `quarter`, `quarters`, `q`, `qtr`, `qtrs`
    -   MONTH: `month`, `months`, `mm`, `mon`, `mons`
    -   WEEK: `week`, `weeks`, `weekofyear`, `w`, `wk`, `woy`, `wy`
    -   DAY: `day`, `days`, `dayofmonth`, `d`, `dd`
    -   HOUR: `hour`, `hours`, `hrs`, `h`, `hr`, `hrs`
    -   MINUTE: `minute`, `minutes`, `m`, `mi`, `min`, `mins`
    -   SECOND: `second`, `seconds`, `s`, `sec`, `secs`
    -   MILLISECOND: `millisecond`, `milliseconds`, `ms`, `msecs`
    -   MICROSECOND: `microsecond`, `microseconds`, `us`, `usec`
    -   NANOSECOND: `nanosecond`, `nanoseconds`, `nanosec`, `nsec`, `nsecs`, `nsecond`, `ns`, `nanonsecs`

    Supported with timezone-aware data.

#### MICROSECOND
-   `#!sql MICROSECOND(timestamp_val)`

    Equivalent to `#!sql EXTRACT(MICROSECOND from timestamp_val)`

#### SECOND
-   `#!sql SECOND(timestamp_val)`

    Equivalent to `#!sql EXTRACT(SECOND from timestamp_val)`

#### MINUTE
-   `#!sql MINUTE(timestamp_val)`

    Equivalent to `#!sql EXTRACT(MINUTE from timestamp_val)`

#### HOUR
-   `#!sql HOUR(timestamp_val)`

    Equivalent to `#!sql EXTRACT(HOUR from timestamp_val)`

#### WEEK
-   `#!sql WEEK(timestamp_val)`

    Equivalent to `#!sql EXTRACT(WEEK from timestamp_val)`

#### WEEKOFYEAR
-   `#!sql WEEKOFYEAR(timestamp_val)`

    Equivalent to `#!sql EXTRACT(WEEK from timestamp_val)`

#### MONTH
-   `#!sql MONTH(timestamp_val)`

    Equivalent to `#!sql EXTRACT(MONTH from timestamp_val)`

#### QUARTER
-   `#!sql QUARTER(timestamp_val)`

    Equivalent to `#!sql EXTRACT(QUARTER from timestamp_val)`

#### YEAR
-   `#!sql YEAR(timestamp_val)`

    Equivalent to `#!sql EXTRACT(YEAR from timestamp_val)`

#### WEEKISO
-   `#!sql WEEKISO(timestamp_val)`

    Computes the ISO week for the provided timestamp value.

#### YEAROFWEEKISO
-   `#!sql YEAROFWEEKISO(timestamp_val)`

    Computes the ISO year for the provided timestamp value.

#### MAKEDATE
-   `#!sql MAKEDATE(integer_years_val, integer_days_val)`

    Computes a timestamp value that is the specified number of days after
    the specified year.

#### DAYNAME
-   `#!sql DAYNAME(timestamp_val)`

    Computes the 3 letter abreviation for the day of the timestamp value.

#### MONTHNAME
-   `#!sql MONTHNAME(timestamp_val)`

    Computes the 3 letter abreviation for the month of the timestamp value.

#### MONTH_NAME
-   `#!sql MONTH_NAME(timestamp_val)`

    Computes the 3 letter abreviation for the month of the timestamp value.

#### TO_DAYS
-   `#!sql TO_DAYS(timestamp_val)`

    Computes the difference in days between the input timestamp, and year
    0 of the Gregorian calendar

#### TO_SECONDS
-   `#!sql TO_SECONDS(timestamp_val)`

    Computes the number of seconds since year 0 of the Gregorian calendar

#### FROM_DAYS
-   `#!sql FROM_DAYS(n)`

    Returns a timestamp values that is n days after year 0 of the
    Gregorian calendar

#### UNIX_TIMESTAMP
-   `#!sql UNIX_TIMESTAMP()`

    Computes the number of seconds since the unix epoch

#### FROM_UNIXTIME
-   `#!sql FROM_UNIXTIME(n)`

    Returns a Timestamp value that is n seconds after the unix epoch

#### ADDDATE
-   `#!sql ADDDATE(timestamp_val, interval)`

    Same as `#!sql DATE_ADD`

#### SUBDATE
-   `#!sql SUBDATE(timestamp_val, interval)`

    Same as `#!sql DATE_SUB`

#### TIMESTAMPDIFF
-   `#!sql TIMESTAMPDIFF(unit, timestamp_val1, timestamp_val2)`

    Returns the amount of time that has passed since `timestamp_val1` until
    `timestamp_val2` in terms of the unit specified, ignoring all smaller units.
    E.g., December 31 of 2020 and January 1 of 2021 count as 1 year apart.

    !!! note
        For all units larger than `#!sql NANOSECOND`, the output type is `#!sql INTEGER`
        instead of `#!sql BIGINT`, so any difference values that cannot be stored as
        signed 32-bit integers might not be returned correct.

#### WEEKDAY
-   `#!sql WEEKDAY(timestamp_val)`

    Returns the weekday number for timestamp_val.

    !!! note
        `Monday = 0`, `Sunday=6`

#### YEARWEEK
-   `#!sql YEARWEEK(timestamp_val)`

    Returns the year and week number for the provided timestamp_val
    concatenated as a single number. For example:
    ```sql
    YEARWEEK(TIMESTAMP '2021-08-30::00:00:00')
    202135
    ```

#### LAST_DAY
-   `#!sql LAST_DAY(timestamp_val)`

    Given a timestamp value, returns a timestamp value that is the last
    day in the same month as timestamp_val.

#### UTC_TIMESTAMP
-   `#!sql UTC_TIMESTAMP()`

    Returns the current UTC date and time as a timestamp value.

#### SYSDATE
-   `SYSDATE()`

    Equivalent to `UTC_TIMESTAMP`

#### UTC_DATE
-   `#!sql UTC_DATE()`

    Returns the current UTC date as a Timestamp value.


###  String Functions

BodoSQL currently supports the following string functions:

#### LOWER
-   `#!sql LOWER(str)`

    Converts the string scalar/column to lower case.

#### LCASE
-   `#!sql LCASE(str)`

    Same as `#!sql LOWER`.

#### UPPER
-   `#!sql UPPER(str)`

    Converts the string scalar/column to upper case.

#### UCASE
-   `#!sql UCASE(str)`

    Same as `#!sql UPPER`.

#### CONCAT
-   `#!sql CONCAT(str_0, str_1, ...)`

    Concatenates the strings together. Requires at least one
    argument.

#### CONCAT_WS
-   `#!sql CONCAT_WS(str_separator, str_0, str_1, ...)`

    Concatenates the strings together, with the specified
    separator. Requires at least two arguments.

#### SUBSTRING
-   `#!sql SUBSTRING(str, start_index, len)`

    Takes a substring of the specified string, starting at the
    specified index, of the specified length. `start_index = 1`
    specifies the first character of the string, `start_index =
    -1` specifies the last character of the string. `start_index
    = 0` causes the function to return empty string. If
    `start_index` is positive and greater than the length of the
    string, returns an empty string. If `start_index` is
    negative, and has an absolute value greater than the
    length of the string, the behavior is equivalent to
    `start_index = 1`.

    For example:

    ```sql
    SUBSTRING('hello world', 1, 5) =='hello'
    SUBSTRING('hello world', -5, 7) =='world'
    SUBSTRING('hello world', -20, 8) =='hello wo'
    SUBSTRING('hello world', 0, 10) ==''
    ```
#### MID
-   `#!sql MID(str, start_index, len)`

    Equivalent to `#!sql SUBSTRING`

#### SUBSTR
-   `#!sql SUBSTR(str, start_index, len)`

    Equivalent to `#!sql SUBSTRING`

#### LEFT
-   `#!sql LEFT(str, n)`

    Takes a substring of the specified string consisting of
    the leftmost n characters

#### RIGHT
-   `#!sql RIGHT(str, n)`

    Takes a substring of the specified string consisting of
    the rightmost n characters

#### REPEAT
-   `#!sql REPEAT(str, len)`

    Extends the specified string to the specified length by
    repeating the string. Will truncate the string If the
    string's length is less than the len argument

    For example:

    ```sql
    REPEAT('abc', 7) =='abcabca'
    REPEAT('hello world', 5) =='hello'
    ```


#### STRCMP
-   `#!sql STRCMP(str1, str2)`

    Compares the two strings lexicographically. If `str1 > str2`,
    return 1. If `str1 < str2`, returns -1. If `str1 == str2`,
    returns 0.

#### REVERSE
-   `#!sql REVERSE(str)`

    Returns the reversed string.

#### ORD
-   `#!sql ORD(str)`

    Returns the integer value of the unicode representation of
    the first character of the input string. returns 0 when
    passed the empty string

#### CHAR
-   `#!sql CHAR(int)`

    Returns the character of the corresponding unicode value.
    Currently only supported for ASCII characters (0 to 127,
    inclusive)

#### SPACE
-   `#!sql SPACE(int)`

    Returns a string containing the specified number of
    spaces.

#### LTRIM
-   `#!sql LTRIM(str[, chars])`

    Removes leading characters from a string column/literal str.
    These characters are specified by chars or are whitespace.

#### RTRIM
-   `#!sql RTRIM(str[, chars])`

    Removes trailing characters from a string column/literal str.
    These characters are specified by chars or are whitespace.

#### TRIM
-   `#!sql TRIM(str[, chars])`

    Returns the input string, will remove all spaces from the
    left and right of the string

#### SUBSTRING_INDEX
-   `#!sql SUBSTRING_INDEX(str, delimiter_str, n)`

    Returns a substring of the input string, which contains
    all characters that occur before n occurrences of the
    delimiter string. if n is negative, it will return all
    characters that occur after the last n occurrences of the
    delimiter string. If `num_occurrences` is 0, it will return
    the empty string

    For example:
    ```sql
    SUBSTRING_INDEX('1,2,3,4,5', ',', 2) =='1,2'
    SUBSTRING_INDEX('1,2,3,4,5', ',', -2) =='4,5'
    SUBSTRING_INDEX('1,2,3,4,5', ',', 0) ==''
    ```


#### LPAD
-   `#!sql LPAD(string, len, padstring)`

    Extends the input string to the specified length, by
    appending copies of the padstring to the left of the
    string. If the input string's length is less than the len
    argument, it will truncate the input string.

    For example:
    ```sql
    LPAD('hello', 10, 'abc') =='abcabhello'
    LPAD('hello', 1, 'abc') =='h'
    ```

#### RPAD
-   `#!sql RPAD(string, len, padstring)`

    Extends the input string to the specified length, by
    appending copies of the padstring to the right of the
    string. If the input string's length is less than the len
    argument, it will truncate the input string.

    For example:
    ```sql
    RPAD('hello', 10, 'abc') =='helloabcab'
    RPAD('hello', 1, 'abc') =='h'
    ```


#### REPLACE
-   `#!sql REPLACE(base_string, substring_to_remove, string_to_substitute)`

    Replaces all occurrences of the specified substring with
    the substitute string.

    For example:
    ```sql
    REPLACE('hello world', 'hello' 'hi') =='hi world'
    ```


#### LENGTH
-   `#!sql LENGTH(string)`

    Returns the number of characters in the given string.


#### EDITDISTANCE
-   `#!sql EDITDISTANCE(string0, string1[, max_distance])`

    Returns the minimum edit distance between `#!sql string0` and `#!sql string1`
    according to Levenshtein distance. Optionally accepts a third
    argument specifying a maximum distance value. If the minimum
    edit distance between the two strings exceeds this value, then
    this value is returned instead. If it is negative, zero
    is returned.


#### JAROWINKLER_SIMILARITY
-   `#!sql JAROWINKLER_SIMILARITY(string0, string1)`

    Computes the Jaro-Winkler similarity between `#!sql string0`
    and `#!sql string1` as an integer between 0 and 100 (with 0
    being no similarity and 100 being an exact match). The computation
    is not case-sensitive, but is sensitive to spaces or formatting
    characters. A scaling factor of 0.1 is used for the computation.
    For the definition of Jaro-Winkler similarity, [see here](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance).



#### SPLIT_PART
-   `#!sql SPLIT_PART(source, delimiter, part)`

    Returns the substring of the source between certain occurrence of
    the delimiter string, the occurrence being specified by the part.
    I.e. if part=1, returns the substring before the first occurrence,
    and if part=2, returns the substring between the first and second
    occurrence. Zero is treated like 1. Negative indices are allowed.
    If the delimiter is empty, the source is treated like a single token.
    If the part is out of bounds, '' is returned.


#### STRTOK
-   `#!sql STRTOK(source[, delimiter[, part]])`

    Tokenizes the source string by occurrences of any character in the
    delimiter string and returns the occurrence specified by the part.
    I.e. if part=1, returns the substring before the first occurrence,
    and if part=2, returns the substring between the first and second
    occurrence. Zero and negative indices are not allowed. Empty tokens
    are always skipped in favor of the next non-empty token. In any
    case where the only possible output is '', the output is `NULL`.
    The delimiter is optional and defaults to ' '. The part is optional
    and defaults to 1.


#### POSITION
-   `#!sql POSITION(str1, str2)`

    Returns the 1-indexed location where `str1` first occurs in `str2`, or 0 if
    there is no occurrences of `str1` in `str2`.

    !!! note
        BodoSQL oes not currently support alternate syntax `#!sql POSITION(str1, str2)`, or binary data.


#### CHARINDEX
-   `#!sql CHARINDEX(str1, str2[, start_position])`

    Equivalent to `#!sql POSITION(str1, str2)` when 2 arguments are provided. When the
    optional third argument is provided, it only starts searching at that index.

    !!! note
        Not currently supported on binary data.


#### STARTSWITH
-   `#!sql STARTSWITH(str1, str2)`

    Returns whether `str2` is a prefix of `str1`.


#### ENDSWITH
-   `#!sql ENDSWITH(str1, str2)`

    Returns whether `str2` is a suffix of `str1`.


#### INSERT
-   `#!sql INSERT(str1, pos, len, str2)`

    Inserts `str2` into `str1` at position `pos` (1-indexed), replacing
    the first `len` characters after `pos` in the process. If `len` is zero,
    inserts `str2` into `str1` without deleting any characters. If `pos` is one,
    prepends `str2` to `str1`. If `pos` is larger than the length of `str1`, appends
    `str2` to `str1`.

    !!! note
        Behavior when `pos` or `len` are negative is not well-defined at this time.


#### SHA2
-   `#!sql SHA2(msg[, digest_size])`

    Encodes the `msg` string using the `SHA-2` algorithm with the specified
    digest size (only values supported are, 224, 256, 384 and 512). Outputs
    the result as a hex-encoded string.


#### SHA2_HEX
-   `#!sql SHA2_HEX(msg[, digest_size])`

    Equivalent to `#!sql SHA2(msg[, digest_size])`


#### MD5
-   `#!sql MD5(msg)`

    Encodes the `msg` string using the `MD5` algorithm. Outputs the
    result as a hex-encoded string.


#### MD5_HEX
-   `#!sql MD5_HEX(msg)`

    Equivalent to `#!sql MD5_HEX(msg)`


#### HEX_ENCODE
-   `#!sql HEX_ENCODE(msg[, case])`

    Encodes the `msg` string into a string using the hex encoding scheme as if
    it were binary data (or directly encodes binary data). If `#!sql case`
    (default one) is zero then the alphabetical hex characters are lowercase,
    if it is one then they are uppercase.
    [See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/hex_encode).


#### HEX_DECODE_STRING
-   `#!sql HEX_DECODE_STRING(msg)`

   Reverses the process of calling `#!sql HEX_ENCODE` on a string with either capitalization.
   Raises an exception if the string is malformed in any way.
    [See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/hex_decode_string).


#### TRY_HEX_DECODE_STRING
-   `#!sql TRY_HEX_DECODE_STRING(msg)`

    Equivalent to `#!sql HEX_DECODE_STRING` except that it will return null instead of raising
    an exception if the string is malformed in any way.


#### HEX_DECODE_BINARY
-   `#!sql HEX_DECODE_BINARY(msg)`

    The same as `#!sql HEX_DECODE_STRING` except that the output is binary instead of a string.



#### TRY_HEX_DECODE_BINARY
-   `#!sql TRY_HEX_DECODE_BINARY(msg)`

    Equivalent to `#!sql HEX_DECODE_BINARY` except that it will return null instead of raising
    an exception if the string is malformed in any way.


#### BASE64_ENCODE
-   `#!sql BASE64_ENCODE(msg[, max_line_length[, alphabet]])`

    Encodes the `msg` string into a string using the base64 encoding scheme as if
    it were binary data (or directly encodes binary data). If `#!sql max_line_length`
    (default zero) is greater than zero, then newline characters will be inserted
    after that many characters to effectively add "text wrapping". If `#!sql alphabet`
    is provided, it specifies substitutes for the usual encoding characters for
    index 62, index 63, and the padding character.
    [See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/base64_encode).


#### BASE64_DECODE_STRING
-   `#!sql BASE64_DECODE_STRING(msg[, alphabet])`

    Reverses the process of calling `#!sql BASE64_ENCODE` on a string with the given alphabet,
    ignoring any newline characters produced by the `#!sql max_line_length` argument. Raises an
    exception if the string is malformed in any way.
    [See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/base64_decode_string).


#### TRY_BASE64_DECODE_STRING
-   `#!sql TRY_BASE64_DECODE_STRING(msg[, alphabet])`

    Equivalent to `#!sql BASE64_DECODE_STRING` except that it will return null instead of raising
    an exception if the string is malformed in any way.


#### BASE64_DECODE_BINARY
-   `#!sql BASE64_DECODE_BINARY(msg[, alphabet])`

    The same as `#!sql BASE64_DECODE_STRING` except that the output is binary instead of a string.



#### TRY_BASE64_DECODE_BINARY
-   `#!sql TRY_BASE64_DECODE_BINARY(msg[, alphabet])`

    Equivalent to `#!sql BASE64_DECODE_BINARY` except that it will return null instead of raising
    an exception if the string is malformed in any way.


###  Regex Functions

BodoSQL currently uses Python's regular expression library via the `re`
module. Although this may be subject to change, it means that there are
several deviations from the behavior of Snowflake's regular expression
functions [(see here for snowflake documentation)](https://docs.snowflake.com/en/sql-reference/functions-regexp.html).
The key points and major deviations are noted below:

* Snowflake uses a superset of the POSIX ERE regular expression syntax. This means that BodoSQL can utilize several syntactic forms of regular expressions that Snowflake cannot [(see here for Python re documentation)](https://docs.python.org/3/library/re.html). However, there are several features that POSIX ERE has that Python's `re` does not:

   - POSIX character classes [(see here for a full list)](https://en.wikipedia.org/wiki/Regular_expression#Character_classes). BodoSQL does support these as macros for character sets. In other words, `[[:lower:]]` is transformed into `[a-z]`. However, this form of replacement cannot be escaped. Additionally, any character classes that are supposed to include the null terminator `\x00` instead start at `\x01`

   - Equivalence classes (not supported by BodoSQL).

   - Returning the longest match when using alternation patterns (BodoSQL returns the leftmost match).

* The regex functions can optionally take in a flag argument. The flag is a string whose characters control how matches to patterns occur. The following characters have meaning when contained in the flag string:

   - `'c'`: case-sensitive matching (the default behavior)
   - `'i'`: case-insensitive matching (if both 'c' and 'i' are provided, whichever one occurs last is used)
   - `'m'`: allows anchor patterns to interact with the start/end of each line, not just the start/end of the entire string.
   - `'s'`: allows the `.` metacharacter to capture newline characters
   - `'e'`: see `REGEXP_SUBSTR`/`REGEXP_INSTR`

* Currently, BodoSQL supports the lazy `?` operator whereas Snowflake does not. So for example, in Snowflake, the pattern ``(.*?),'` would match with as many characters as possible so long as the last character was a comma. However, in BodoSQL, the match would end as soon as the first comma.

* Currently, BodoSQL supports the following regexp features which should crash when done in Snowflake: `(?...)`, `\A`, `\Z`, `\1`, `\2`, `\3`, etc.

* Currently, BodoSQL requires the pattern argument and the flag argument (if provided) to be string literals as opposed to columns or expressions.

* Currently, extra backslashes may be required to escape certain characters if they have meaning in Python. The amount of backslashes required to properly escape a character depends on the usage.

* All matches are non-overlapping.

* If any of the numeric arguments are zero or negative, or the `group_num` argument is out of bounds, an error is raised. The only exception is `#!sql REGEXP_REPLACE`, which allows its occurrence argument to be zero.

BodoSQL currently supports the following regex functions:

#### REGEXP_LIKE
-   `#!sql REGEXP_LIKE(str, pattern[, flag])`

    Returns `true` if the entire string matches with the pattern.
    If `flag` is not provided, `''` is used.

    If the pattern is empty, then `true` is returned if
    the string is also empty.

    For example:

    - 2 arguments: Returns `true` if `A` is a 5-character string where the first character is an a,
    the last character is a z, and the middle 3 characters are also lowercase characters (case-sensitive).
    ```sql
    SELECT REGEXP_LIKE(A, 'a[a-z]{3}z')
    ```

    - 3 arguments: Returns `true` if `A` starts with the letters `'THE'` (case-insensitive).
    ```sql
    SELECT REGEXP_LIKE(A, 'THE.*', 'i')
    ```


#### REGEXP_COUNT
-   `#!sql REGEXP_COUNT(str, pattern[, position[, flag]])`

    Returns the number of times the string contains matches
    to the pattern, starting at the location specified
    by the `position` argument (with 1-indexing).
    If `position` is not provided, `1` is used.
    If `flag` is not provided, `''` is used.

    If the pattern is empty, 0 is returned.

    For example:

    - 2 arguments: Returns the number of times that any letters occur in `A`.
    ```sql
    SELECT REGEXP_COUNT(A, '[[:alpha:]]')
    ```

    - 3 arguments: Returns the number of times that any digit characters occur in `A`, not including
    the first 5 characters.
    ```sql
    SELECT REGEXP_COUNT(A, '\d', 6)
    ```

    - 4 arguments: Returns the number of times that a substring occurs in `A` that contains two
    ones with any character (including newlines) in between.
    ```sql
    SELECT REGEXP_COUNT(A, '1.1', 1, 's')
    ```


#### REGEXP_REPLACE
-   `#!sql REGEXP_REPLACE(str, pattern[, replacement[, position[, occurrence[, flag]]]])`

    Returns the version of the inputted string where each
    match to the pattern is replaced by the replacement string,
    starting at the location specified by the `position` argument
    (with 1-indexing). The occurrence argument specifies which
    match to replace, where 0 means replace all occurrences. If
    `replacement` is not provided, `''` is used. If `position` is
    not provided, `1` is used. If `occurrence` is not provided,
    `0` is used. If `flag` is not provided, `''` is used.

    If there are an insufficient number of matches, or the pattern is empty,
    the original string is returned.

    !!! note
        back-references in the replacement pattern are supported,
        but may require additional backslashes to work correctly.

    For example:

    - 2 arguments: Deletes all whitespace in `A`.
    ```sql
    SELECT REGEXP_REPLACE(A, '[[:space:]]')
    ```

    - 3 arguments: Replaces all occurrences of `'hate'` in `A` with `'love'` (case-sensitive).
    ```sql
    SELECT REGEXP_REPLACE(A, 'hate', 'love')
    ```

    - 4 arguments: Replaces all occurrences of two consecutive digits in `A` with the same two
    digits reversed, excluding the first 2 characters.
    ```sql
    SELECT REGEXP_REPLACE(A, '(\d)(\d)', '\\\\2\\\\1', 3)
    ```

    - 5 arguments: Replaces the first character in `A` with an underscore.
    ```sql
    SELECT REGEXP_REPLACE(A, '^.', '_', 1, 2)
    ```

    - 6 arguments: Removes the first and last word from each line of `A` that contains
    at least 3 words.
    ```sql
    SELECT REGEXP_REPLACE(A, '^\w+ (.*) \w+$', '\\\\1', 0, 1, 'm')
    ```


#### REGEXP_SUBSTR
-   `#!sql REGEXP_SUBSTR(str, pattern[, position[, occurrence[, flag[, group_num]]]])`

    Returns the substring of the original string that caused a
    match with the pattern, starting at the location specified
    by the `position` argument (with 1-indexing). The occurrence argument
    specifies which match to extract (with 1-indexing). If `position` is
    not provided, `1` is used. If `occurrence` is not provided,
    `1` is used. If `flag` is not provided, `''` is used. If `group_num`
    is not provided, and `flag` contains `'e`', `1` is used. If `group_num` is provided but the
    flag does not contain `e`, then it behaves as if it did. If the flag does contain `e`,
    then one of the subgroups of the match is returned instead of the entire match. The
    subgroup returned corresponds to the `group_num` argument
    (with 1-indexing).

    If there are an insufficient number of matches, or the pattern is empty,
    `NULL` is returned.

    For example:

    - 2 arguments: Returns the first number that occurs inside of `A`.
    ```sql
    SELECT REGEXP_SUBSTR(A, '\d+')
    ```

    - 3 arguments: Returns the first punctuation symbol that occurs inside of `A`, excluding the first 10 characters.
    ```sql
    SELECT REGEXP_SUBSTR(A, '[[:punct:]]', 11)
    ```

    - 4 arguments: Returns the fourth occurrence of two consecutive lowercase vowels in `A`.
    ```sql
    SELECT REGEXP_SUBSTR(A, '[aeiou]{2}', 1, 4)
    ```

    - 5 arguments: Returns the first 3+ character substring of `A` that starts with and ends with a vowel (case-insensitive, and
    it can contain newline characters).
    ```sql
    SELECT REGEXP_SUBSTR(A, '[aeiou].+[aeiou]', 1, 1, 'im')
    ```

    - 6 arguments: Looks for third occurrence in `A` of a number followed by a colon, a space, and a word
    that starts with `'a'` (case-sensitive) and returns the word that starts with `'a'`.
    ```sql
    SELECT REGEXP_SUBSTR(A, '(\d+): (a\w+)', 1, 3, 'e', 2)
    ```


#### REGEXP_INSTR
-   `#!sql REGEXP_INSTR(str, pattern[, position[, occurrence[, option[, flag[, group_num]]]]])`

    Returns the location within the original string that caused a
    match with the pattern, starting at the location specified
    by the `position` argument (with 1-indexing). The occurrence argument
    specifies which match to extract (with 1-indexing). The option argument
    specifies whether to return the start of the match (if `0`) or the first
    location after the end of the match (if `1`). If `position` is
    not provided, `1` is used. If `occurrence` is not provided,
    `1` is used. If `option` is not provided, `0` is used. If `flag` is not
    provided, `''` is used. If `group_num` is not provided, and `flag` contains `'e`', `1` is used.
    If `group_num` is provided but the flag does not contain `e`, then
    it behaves as if it did. If the flag does contain `e`, then the location of one of
    the subgroups of the match is returned instead of the location of the
    entire match. The subgroup returned corresponds to the `group_num` argument
    (with 1-indexing).

    If there are an insufficient number of matches, or the pattern is empty,
    `0` is returned.

    - 2 arguments: Returns the index of the first `'#'` in `A`.
    ```sql
    SELECT REGEXP_INSTR(A, '#')
    ```

    - 3 arguments: Returns the starting index of the first occurrence of 3 consecutive digits in `A`,
    excluding the first 3 characters.
     ```sql
    SELECT REGEXP_INSTR(A, '\d{3}', 4)
    ```

    - 4 arguments: Returns the starting index of the 9th word sandwiched between angle brackets in `A`.
    ```sql
    SELECT REGEXP_INSTR(A, '<\w+>', 1, 9)
    ```

    - 5 arguments: Returns the ending index of the first substring of `A` that starts
    and ends with non-ascii characters.
    ```sql
    SELECT REGEXP_INSTR(A, '[^[:ascii:]].*[^[:ascii:]]', 1, 1, 1)
    ```

    - 6 arguments: Returns the starting index of the second line of `A` that begins with an uppercase vowel.
    ```sql
    SELECT REGEXP_INSTR(A, '^[AEIOU].*', 1, 2, 0, 'm')
    ```

    - 7 arguments: Looks for the first substring of `A` that has the format of a name in a phonebook (i.e. `Lastname, Firstname`)
    and returns the starting index of the first name.
    ```sql
    SELECT REGEXP_INSTR(A, '([[:upper]][[:lower:]]+), ([[:upper]][[:lower:]]+)', 1, 1, 0, 'e', 2)
    ```


###  JSON Functions


BodoSQL currently supports the following JSON functions:


#### OBJECT_CONSTRUCT_KEEP_NULL
-   `#!sql OBJECT_CONSTRUCT_KEEP_NULL(key1, value1[, key2, value2, ...])`

    Takes in a variable number of key-value pairs and combines them
    into JSON data. BodoSQL currently requires all `key` arguments to
    be string literals.

    The full Snowflake specification: https://docs.snowflake.com/en/sql-reference/functions/object_construct_keep_null.html

    BodoSQL supports the syntactic sugar `#!sql OBJECT_CONSTRUCT_KEEP_NULL(*)`
    which indicates that all columns should be used as key-value pairs, where
    the column is the value and its column name is the key. For example, if we have
    the table `T` as defined below:

    | First    | Middle   | Last         |
    |----------|----------|--------------|
    | "George" | NULL     | "WASHINGTON" |
    | "John"   | "Quincy" | "Adams"      |
    | "Lyndon" | "Baines" | "Johnson"    |
    | "James"  | NULL     | "Madison"    |

    Then `SELECT OBJECT_CONSTRUCT(*) as name FROM T` returns the following table:

    | name                                                      |
    |-----------------------------------------------------------|
    | {"First": "George", "Middle": NULL, "Last": "Washington"} |
    | {"First": "John", "Middle": "Quincy", "Last": "Adams"}    |
    | {"First": "Lyndon", "Middle":"Baines", "Last": "Johnson"} |
    | {"First": "Thomas", "Middle": NULL, "Last": "Jefferson"}  |


#### OBJECT_CONSTRUCT
-   `#!sql OBJECT_CONSTRUCT_(key1, value1[, key2, value2, ...])`

    The same as `#!sql OBJECT_CONSTRUCT_KEEP_NULL` except that for any rows where any input value 
    (e.g. `value1`, `value2`, ...) is null have that key-value pair dropped from the row's final JSON output.

    !!! note: BodoSQL only supports this function under narrow conditions where all of the values
    are either of the same type or of easily reconciled types.

    The full Snowflake specification: https://docs.snowflake.com/en/sql-reference/functions/object_construct.html

    BodoSQL supports the syntactic sugar `#!sql OBJECT_CONSTRUCT(*)`
    which indicates that all columns should be used as key-value pairs, where
    the column is the value and its column name is the key. For example, if we have
    the table `T` as defined below:

    | First    | Middle   | Last         |
    |----------|----------|--------------|
    | "George" | NULL     | "WASHINGTON" |
    | "John"   | "Quincy" | "Adams"      |
    | "Lyndon" | "Baines" | "Johnson"    |
    | "James"  | NULL     | "Madison"    |

    Then `SELECT OBJECT_CONSTRUCT(*) as name FROM T` returns the following table:

    | name                                                      |
    |-----------------------------------------------------------|
    | {"First": "George", "Last": "Washington"}                 |
    | {"First": "John", "Middle": "Quincy", "Last": "Adams"}    |
    | {"First": "Lyndon", "Middle":"Baines", "Last": "Johnson"} |
    | {"First": "Thomas", "Last": "Jefferson"}                  |



#### OBJECT_KEYS
-   `#!sql OBJECT_KEYS(data)`

    Extracts all of the field names from the JSON object `data` and returns them
    as an array of strings.


#### OBJECT_DELETE
-   `#!sql OBJECT_DELETE(data, key1[, key2, ...])`

    Takes in a column of JSON data and 1+ keys and returns the same JSON data but
    with all of those keys removed.

    !!! note: BodoSQL supports when the keys are passed in as string literals,
    but only sometimes supports when they are passed in as columns of strings.


#### JSON_EXTRACT_PATH_TEXT
-   `#!sql JSON_EXTRACT_PATH_TEXT(data, path)`

    Parses the string `data` as if it were JSON data, then extracts values from
    within (possibly multiple times if the data is nested) using the string `path`.

    Obeys the following specification: https://docs.snowflake.com/en/sql-reference/functions/json_extract_path_text.html


#### GET_PATH
-   `#!sql GET_PATH(data, path_string)`

    Extracts an entry from a semi-structured data expression based on the path string.
    Obeys the specification described here: https://docs.snowflake.com/en/sql-reference/functions/get_path

    !!! note
        Currently only supported under limited conditions where it is possible for the
        planner to push the extraction of JSON fields into the reader or into a filter
        that is pushed down. If a usage of `#sql GET_PATH` causes errors, it may mean
        that it is not currently in one of these limited cases.

    Below are some valid examples of usage and alternative syntax:

    - `#!sql SELECT product_id FROM products WHERE GET_PATH(appearance, 'color') = 'RED'`

    - `#!sql SELECT product_id FROM products WHERE appearance:color = 'RED'`

    - `#!sql SELECT CONCAT_WS(' ', GET_PATH(info, 'first'), GET_PATH(info, 'last')) FROM phone_book'`
    Note that this case the query creates an implicit cast to `varchar`.

    - `#!sql SELECT CONCAT_WS(' ', info:first, info:last) FROM phone_book'`
    Note that this case the query creates an implicit cast to `varchar`.

###   Control Flow Functions

#### DECODE
-   `#!sql DECODE(Arg0, Arg1, Arg2, ...)`

    When `Arg0` is `Arg1`, outputs `Arg2`. When `Arg0` is `Arg3`,
    outputs `Arg4`. Repeats until it runs out of pairs of arguments.
    At this point, if there is one remaining argument, this is used
    as a default value. If not, then the output is `NULL`.

    !!! note
        Treats `NULL` as a literal value that can be matched on.

    Therefore, the following:

    ```sql
    DECODE(A, NULL, 0, 'x', 1, 'y', 2, 'z', 3, -1)
    ```

    Is logically equivalent to:

    ```sql
    CASE WHEN A IS NULL THEN 0
         WHEN A = 'x' THEN 1
         WHEN A = 'y' THEN 2
         WHEN A = 'z' THEN 3
         ELSE -1 END
    ```

#### EQUAL_NULL
-   `#!sql EQUAL_NULL(A, B)`

    Returns true if A and B are both `NULL`, or both non-null and
    equal to each other.


#### IF
-   `#!sql IF(Cond, TrueValue, FalseValue)`

    Returns `TrueValue` if cond is true, and `FalseValue` if cond is
    false. Logically equivalent to:

    ```sql
    CASE WHEN Cond THEN TrueValue ELSE FalseValue END
    ```


#### IFF
-   `#!sql IFF(Cond, TrueValue, FalseValue)`

    Equivalent to `#!sql IF`


#### IFNULL
-   `#!sql IFNULL(Arg0, Arg1)`

    Returns `Arg1` if `Arg0` is `null`, and otherwise returns `Arg1`. If
    arguments do not have the same type, BodoSQL will attempt
    to cast them all to a common type, which is currently
    undefined behavior.

#### ZEROIFNULL
-   `#!sql ZEROIFNULL(Arg0, Arg1)`

    Equivalent to `#!sql IFNULL(Arg0, 0)`


#### NVL
-   `#!sql NVL(Arg0, Arg1)`

    Equivalent to `#!sql IFNULL`


#### NVL2
-   `#!sql NVL2(Arg0, Arg1, Arg2)`

    Equivalent to `#!sql IF(Arg0 IS NOT NULL, Arg1, Arg2)`


#### NULLIF
-   `#!sql NULLIF(Arg0, Arg1)`

    Returns `null` if the `Arg0` evaluates to true, and otherwise
    returns `Arg1`


#### NULLIFZERO
-   `#!sql NULLIFZERO(Arg0)`

    Equivalent to `#!sql NULLIF(Arg0, 0)`


#### COALESCE
-   `#!sql COALESCE(A, B, C, ...)`

    Returns the first non-`NULL` argument, or `NULL` if no non-`NULL`
    argument is found. Requires at least two arguments. If
    Arguments do not have the same type, BodoSQL will attempt
    to cast them to a common data type, which is currently
    undefined behavior.


### Array Functions
Bodo currently supports the following functions that operate on columns of arrays:

#### GET
-   `#!sql GET(arr, idx)`
-   `#!sql arr[idx]`

    Returns the element found at the specified index in the array. Inexing is 0 based, not 1 based. Returns NULL if the index is outside of the boundaries of the array.

#### TO_ARRAY
-   `#!sql TO_ARRAY(arr)`

    Converts the input expression to a single-element array containing this value. If the input
    is an ARRAY, or VARIANT containing an array value, the result is unchanged. Returns `NULL`
    for `NULL` or a JSON null input.


#### ARRAY_TO_STRING
-   `#!sql ARRAY_TO_STRING(arr, sep)`

    Converted the input array `arr` to a string by casting all values to strings (using `TO_VARCHAR`)
    and concatenating them (using `sep` to separate the elements).


#### ARRAY_COMPACT
-   `#!sql ARRAY_COMPACT(arr)`

    Returns a compacted array with missing and null values removed from `arr`, effectively
    converting sparse arrays into dense arrays. Return `NULL` when `arr` is `NULL`.


#### ARRAY_CONTAINS
-   `#!sql ARRAY_CONTAINS(elem, arr)`

    Returns true if `elem` is an element of `arr`, or `NULL` if `arr` is `NULL`. The input
    `elem` can be `NULL`, in which case the funciton will check if `arr` contains `NULL`.


#### ARRAY_CONSTRUCT
-   `#!sql ARRAY_CONSTRUCT(A, B, C, ...)`

    Takes in a variable number of arguments and produces an array containing
    all of those values (including any null values).

    Note: Snowflake allows any number of arguments (even zero arguments) of any
    type. BodoSQL currently requires 1+ arguments, and requires all arguments
    to be easily reconciled into a common type.


#### ARRAY_EXCEPT
-   `#!sql ARRAY_EXCEPT(A, B)`

    Takes in two arrays and returns a copy of the first array but with all
    of the elements from the second array dropped. If an element appears in
    the first array more than once, that element is only dropped as many
    times as it appears in the second array. For instance, if the
    first array contains three 1s and four 6s, and the second array
    contains two 1s and one 6, then the output will have one 1 and
    three 6s.


#### ARRAY_INTERSECTION
-   `#!sql ARRAY_INTERSECTION(A, B)`

    Takes in two arrays and returns an arary of all the elements from the
    first array that also appear in the second. If an element appears in
    either array more than once, that element is kept the minimum of the
    number of times it appears in either array. For instance, if the
    first array contains three 1s and four 6s, and the second array
    contains two 1s and five 6s, then the output will have two 1s and
    three 6s.


#### ARRAY_CAT
-   `#!sql ARRAY_CAT(A, B)`

    Takes in two arrays and returns an arary of all the elements from the
    first array followed by all of the elements in the second array.


#### ARRAYS_OVERLAP
-   `#!sql ARRAYS_OVERLAP(arr0, arr1)`

    Returns true if the two array arguments `arr0` and `arr1` have at least 1 element
    in common (including `NULL`).


#### ARRAYS_POSITION
-   `#!sql ARRAYS_OVERLAP(elem, arr)`

    Returns the index of the first occurrence of `elem` in `arr` (using zero indexing), or
    `NULL` if there are no occurrences. The input `elem` can be `NULL`, in which case the funciton
    will look for the first `NULL` in the array input.


#### ARRAY_SIZE
-   `#!sql ARRAY_SIZE(array)`

    Returns the size of array, size includes inner elements that are `NULL``.
    Returns `NULL` if array is `NULL`.
    [See here for Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/array_size).


#### ARRAY_REMOVE
-   `#!sql ARRAY_REMOVE(array, to_remove)`

    Given a source `array`, returns an array with all elements equal to the specified
    value `to_remove` removed. Returns `NULL` if `array` or `to_remove` is `NULL`.


#### ARRAY_SLICE
-   `#!sql ARRAY_SLICE(arr, from, to)`

    Returns an array constructed from a specified subset of elements of the input array `arr[from:to]`.
    Returns `NULL` if one of `arr`, `from` and `to` is `NULL`.


### Window Functions

Window functions can be used to compute an aggregation across a
row and its surrounding rows. Most window functions have the
following syntax:

```sql
SELECT WINDOW_FN(ARG1, ..., ARGN) OVER (PARTITION BY PARTITION_COLUMN_1, ..., PARTITION_COLUMN_N ORDER BY SORT_COLUMN_1, ..., SORT_COLUMN_N ROWS BETWEEN <LOWER_BOUND> AND <UPPER_BOUND>) FROM table_name
```
The `#!sql ROWS BETWEEN ROWS BETWEEN <LOWER_BOUND> AND <UPPER_BOUND>`
section is used to specify the window over which to compute the
function. A bound can can come before the current row, using `#!sql PRECEDING` or after the current row, using
`#!sql FOLLOWING`. The bounds can be relative (i.e.
`#!sql N PRECEDING` or `#!sql N FOLLOWING`), where `N` is a positive integer,
or they can be absolute (i.e. `#!sql UNBOUNDED PRECEDING` or
`#!sql UNBOUNDED FOLLOWING`).

For example:

```sql
SELECT SUM(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM table1
```
This query computes the sum of every 3 rows, i.e. the sum of a row of interest, its preceding row, and its following row.

In contrast:

```sql
SELECT SUM(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN UNBOUNDED PRECEDING AND 0 FOLLOWING) FROM table1
```
This query computes the cumulative sum over a row and all of its preceding rows.

!!! note
    For most window functions, BodoSQL returns `NULL` if the specified window frame
    is empty or all `NULL`. Exceptions to this behavior are noted.

Window functions perform a series of steps as followed:

1.  Partition the data by `#!sql PARTITION_COLUMN`. This is effectively a groupby operation on `#!sql PARTITION_COLUMN`.
2.  Sort each group as specified by the `#!sql ORDER BY` clause.
3.  Perform the calculation over the specified window, i.e. the newly ordered subset of data.
4.  Shuffle the data back to the original ordering.

For BodoSQL, `#!sql PARTITION BY` is required, but
`#!sql ORDER BY` is optional for most functions and
`#!sql ROWS BETWEEN` is optional for all of them. If
`#!sql ROWS BETWEEN` is not specified then it defaults to either
computing the result over the entire window (if no `#!sql ORDER BY`
clause is specified) or to using the window `#!sql UNBOUNDED PRECEDING TO CURRENT ROW`
(if there is an `#!sql ORDER BY` clause).
!!! note
    `#!sql RANGE BETWEEN` is not currently supported.

Currently, BodoSQL supports the following Window functions:

!!!note
    If a window frame contains `NaN` values, the output may diverge from Snowflake's
    behavior. When a `NaN` value enters a window, any window function that combines
    the results with arithmetic (e.g. `SUM`, `AVG`, `VARIANCE`, etc.) will output
    `NaN` until the `NaN` value has exited the window.


#### COUNT
-   `#!sql COUNT(*)`

    Compute the number of entries in a window, including `NULL`.

#### SUM
-   `#!sql SUM(COLUMN_EXPRESSION)`

    Compute the sum over the window or `NULL` if the window is
    empty.

#### AVG
-   `#!sql AVG(COLUMN_EXPRESSION)`

    Compute the average over the window or `NULL` if the window
    is empty.


#### STDDEV
-   `#!sql STDDEV(COLUMN_EXPRESSION)`

    Compute the standard deviation for a sample over the
    window or `NULL` if the window is empty.

#### STDDEV_POP
-   `#!sql STDDEV_POP(COLUMN_EXPRESSION)`

    Compute the standard deviation for a population over the
    window or `NULL` if the window is empty.

#### VARIANCE
-   `#!sql VARIANCE(COLUMN_EXPRESSION)`

    Compute the variance for a sample over the window or `NULL`
    if the window is empty.

#### VAR_POP
-   `#!sql VAR_POP(COLUMN_EXPRESSION)`

    Compute the variance for a population over the window or
    `NULL` if the window is empty.


#### COVAR_SAMP
-   `#!sql COVAR_SAMP(Y, X)`

    Compute the sample covariance over the window of both inputs, or `NULL` if
    the window is empty.


#### COVAR_POP
-   `#!sql COVAR_POP(Y, X)`

    Compute the population covariance over the window of both inputs, or `NULL` if
    the window is empty.


#### CORR
-   `#!sql CORR(Y, X)`

    Compute the correlation over the window of both inputs, or `NULL` if
    the window is empty. Equivalent to `#!sql COVAR(Y, X) / (STDDEV_POP(Y) * STDDEV_POP(X))`


#### MAX
-   `#!sql MAX(COLUMN_EXPRESSION)`

    Compute the maximum value over the window or `NULL` if the
    window is empty.

#### MIN
-   `#!sql MIN(COLUMN_EXPRESSION)`

    Compute the minimum value over the window or `NULL` if the
    window is empty.

#### COUNT

-   `#!sql COUNT(COLUMN_EXPRESSION)`

    Compute the number of non-`NULL` entries in a window, or zero if the window
    is empty.

#### COUNT_IF

-   `#!sql COUNT_IF(BOOLEAN_COLUMN_EXPRESSION)`

    Compute the number of `true` entries in a boolean column, or zero if the window
    is empty.


#### MEDIAN
-   `#!sql MEDIAN(COLUMN_EXPRESSION)`

    Compute the median over the window, or `NULL` if the window is empty.


#### MODE
-   `MODE(COLUMN_EXPRESSION)`

    Returns the most frequent element in the window, or `NULL` if the window is
    empty.

    !!! note
        In case of a tie, BodoSQL will choose a value arbitrarily based on performance considerations.

#### SKEW
-   `#!sql SKEW(COLUMN_EXPRESSION)`

    Compute the skew over the window, or `NULL` if the window contains fewer
    than 3 non-`NULL` entries.


#### KURTOSIS
-   `#!sql KURTOSIS(COLUMN_EXPRESSION)`

    Compute the skew over the window, or `NULL` if the window contains fewer
    than 4 non-`NULL` entries.

#### BITOR_AGG
-   `#!sql BITOR_AGG`

    Outputs the bitwise OR of every input
    in the window, or `#!sql NULL` if the window has no non-`#!sql NULL` elements.
    Accepts floating point values, integer values, and strings. Strings are interpreted
    directly as numbers, converting to 64-bit floating point numbers.

#### BOOLOR_AGG
-   `#!sql BOOLOR_AGG`

    Outputs `#!sql true` if there is at least 1 non-zero` element in the
    window, or `#!sql NULL` if the window has no non-`#!sql NULL` elements.


#### BOOLAND_AGG
-   `#!sql BOOLAND_AGG`

    Outputs `#!sql true` if every element in the window that is non-`#!sql NULL`
    is also non-zero, or `#!sql NULL` if the window has no non-`#!sql NULL` elements.


#### BOOLXOR_AGG
-   `#!sql BOOLXOR_AGG`

    Outputs `#!sql true` if there is at exactly 1 non-zero element in the
    window, or `#!sql NULL` if the window has no non-`#!sql NULL` elements.


#### LEAD
-   `#!sql LEAD(COLUMN_EXPRESSION, [N], [FILL_VALUE])`

    Returns the row that follows the current row by N. If N
    is not specified, defaults to 1. If FILL_VALUE is not
    specified, defaults to `NULL`. If
    there are fewer than N rows the follow the current row in
    the window, it returns FILL_VALUE. N must be a literal
    non-negative integer if specified. FILL_VALUE must be a
    scalar if specified.

    !!!note
        - At this time Bodo does not support the `#!sql IGNORE NULLS` keyword.
        - This function cannot be used with `#!sql ROWS BETWEEN`.

#### LAG
-   `#!sql LAG(COLUMN_EXPRESSION, [N], [FILL_VALUE])`

    Returns the row that precedes the current row by N. If N
    is not specified, defaults to 1. If FILL_VALUE is not
    specified, defaults to `NULL`. If
    there are fewer than N rows that precede the current row in
    the window, it returns FILL_VALUE. N must be a literal
    non-negative integer if specified. FILL_VALUE must be a
    scalar if specified.

    !!! note
        - At this time BodoSQL does not support the `#!sql IGNORE NULLS` keyword.
        - This function cannot be used with `#!sql ROWS BETWEEN`.

#### FIRST_VALUE
-   `#!sql FIRST_VALUE(COLUMN_EXPRESSION)`

    Select the first value in the window or `NULL` if the window
    is empty.
    Select the first value in the window.

#### LAST_VALUE
-   `#!sql LAST_VALUE(COLUMN_EXPRESSION)`

    Select the last value in the window or `NULL` if the window
    is empty.
    Select the last value in the window.

#### NTH_VALUE
-   `#!sql NTH_VALUE(COLUMN_EXPRESSION, N)`

    Select the Nth value in the window (1-indexed) or `NULL` if
    the window is empty. If N is greater or than the window
    size, this returns `NULL`.
    Select the Nth value in the window (1-indexed). If N is greater or than the
    window size, this returns `NULL`.


#### ANY_VALUE
-   `#!sql ANY_VALUE(COLUMN_EXPRESSION)`

    Select an arbitrary value in the window or `NULL` if the window
    is empty.

    !!! note
        Currently, BodoSQL always selects the first value, but this is subject to change at any time.


#### NTILE
-   `#!sql NTILE(N)`

    Divides the partitioned groups into N buckets based on
    ordering. For example if N=3 and there are 30 rows in a
    partition, the first 10 are assigned 1, the next 10 are
    assigned 2, and the final 10 are assigned 3. In cases where
    the number of rows cannot be evenly divided by the number
    of buckets, the first buckets will have one more value
    than the last bucket. For example, if N=4 and there are
    22 rows in a partition, the first 6 are assigned 1, the
    next 6 are assigned 2, the next 5 are assigned 3, and
    the last 5 are assigned 4.


#### RANK
-   `#!sql RANK()`

    Compute the rank of each row based on the value(s) in the row relative to all value(s) within the partition.
    The rank begins with 1 and increments by one for each succeeding value. Duplicate value(s) will produce
    the same rank, producing gaps in the rank (compare with `#!sql DENSE_RANK`). `#!sql ORDER BY` is required for this function.


#### DENSE_RANK
-   `#!sql DENSE_RANK()`

    Compute the rank of each row based on the value(s) in the row relative to all value(s) within the partition
    without producing gaps in the rank (compare with `#!sql RANK`). The rank begins with 1 and increments by one for each succeeding value.
    Rows with the same value(s) produce the same rank. `#!sql ORDER BY` is required for this function.

!!!note
    To compare `#!sql RANK` and `#!sql DENSE_RANK`, on input array `['a', 'b', 'b', 'c']`, `#!sql RANK` will output `[1, 2, 2, 4]` while `#!sql DENSE_RANK` outputs `[1, 2, 2, 3]`.

#### PERCENT_RANK
-   `#!sql PERCENT_RANK()`

    Compute the percentage ranking of the value(s) in each row based on the value(s) relative to all value(s)
    within the window partition. Ranking calculated using `#!sql RANK()` divided by the number of rows in the window
    partition minus one. Partitions with one row have `#!sql PERCENT_RANK()` of 0. `#!sql ORDER BY` is required for this function.


#### CUME_DIST
-   `#!sql CUME_DIST()`

    Compute the cumulative distribution of the value(s) in each row based on the value(s) relative to all value(s)
    within the window partition. `#!sql ORDER BY` is required for this function.


#### ROW_NUMBER
-   `#!sql ROW_NUMBER()`

    Compute an increasing row number (starting at 1) for each
    row. This function cannot be used with `#!sql ROWS BETWEEN`.

!!! note
    This window function is supported without a partition.


#### CONDITIONAL_TRUE_EVENT
-   `#!sql CONDITIONAL_TRUE_EVENT(BOOLEAN_COLUMN_EXPRESSION)`

    Computes a counter within each partition that starts at zero and increases by 1 each
    time the boolean column's value is `true`. `#!sql ORDER BY` is required for this function.


#### CONDITIONAL_CHANGE_EVENT
-   `#!sql CONDITIONAL_CHANGE_EVENT(COLUMN_EXPRESSION)`

    Computes a counter within each partition that starts at zero and increases by 1 each
    time the value inside the window changes. `NULL` does not count as a new/changed value.
    `#!sql ORDER BY` is required for this function.


#### RATIO_TO_REPORT
-   `#!sql RATIO_TO_REPORT(COLUMN_EXPRESSION)`

    Returns an element in the window frame divided by the sum of all elements in the
    same window frame, or `NULL` if the window frame has a sum of zero. For example,
    if calculating `#!sql RATIO_TO_REPORT` on `[2, 5, NULL, 10, 3]` where the window
    is the entire partition, the answer is `[0.1, 0.25, NULL, 0.5, 0.15]`.


### Casting / Conversion Functions

BodoSQL currently supports the following casting/conversion functions:

#### TO_BOOLEAN
-  `#!sql TO_BOOLEAN(COLUMN_EXPRESSION)`

    Casts the input to a boolean value. If the input is a string, it will be cast to `true` if it is
    `'true'`, `'t'`, `'yes'`, `'y'`, `'1'`, cast to `false` if it is `'false'`, `'f'`, `'no'`, `'n'`, `'0'`,
    and throw an error otherwise.
    If the input is an integer, it will be cast to `true` if it is non-zero and `false` if it is zero.
    If the input is a float, it will be cast to `true` if it is non-zero, `false` if it is zero, and throw an error on other inputs (e.g. `inf`) input. If the input is `NULL`, the output will be `NULL`.

    _Example:_

    We are given `table1` with columns `a` and `b` and `c`
    ```python
    table1 = pd.DataFrame({
        'a': [1.1, 0, 2],
        'b': ['t', 'f', 'YES'],
        'c': [None, 1, 0]
    })
    ```
    upon query
    ```sql
    SELECT
        TO_BOOLEAN(a) AS a,
        TO_BOOLEAN(b) AS b,
        TO_BOOLEAN(c) AS c
    FROM table1;
    ```
    we will get the following output:
    ```
           a      b      c
    0   True   True   <NA>
    1  False  False   True
    2   True   True  False
    ```
    Upon query
    ```sql
    SELECT TO_BOOLEAN('other')
    ```
    we see the following error:
    ```python
    ValueError: invalid value for boolean conversion: string must be one of {'true', 't', 'yes', 'y', 'on', '1'} or {'false', 'f', 'no', 'n', 'off', '0'}
    ```

!!!note
    BodoSQL will read `NaN`s as `NULL` and thus will not produce errors on `NaN` input.

#### TRY_TO_BOOLEAN
-  `#!sql TRY_TO_BOOLEAN(COLUMN_EXPRESSION)`

    This is similar to `#!sql TO_BOOLEAN` except that it will return `NULL` instead of throwing an error invalid inputs.

    _Example:_

    We are given `table1` with columns `a` and `b` and `c`
    ```python
    table1 = pd.DataFrame({
        'a': [1.1, 0, np.inf],
        'b': ['t', 'f', 'YES'],
        'c': [None, 1, 0]
    })
    ```
    upon query
    ```sql
    SELECT
        TRY_TO_BOOLEAN(a) AS a,
        TRY_TO_BOOLEAN(b) AS b,
        TRY_TO_BOOLEAN(c) AS c
    FROM table1;
    ```
    we will get the following output:
    ```
        a      b      c
    0   True   True   <NA>
    1  False  False   True
    2   <NA>   True  False
    ```


#### TO_BINARY
-  `TO_BINARY(COLUMN_EXPRESSION)`

    Casts the input string to binary data. Currently only supports the `HEX` format.
    Raises an exception if the input is not a valid hex string:
    - Must have an even number of characters
    - All characters must be hexedecimal digits (0-9, a-f case insensitive)

    _Example:_

    We are given `table1` with columns `a` and `b`:
    ```python
    table1 = pd.DataFrame({
        'a': ["AB", "626f646f", "4a2F3132"],
        'b': ["ABC", "ZETA", "#fizz"],
    })
    ```
    upon query
    ```sql
    SELECT TO_BINARY(a),
    FROM table1
    ```
    we will get the following output:
    ```
        TO_BINARY(a)
    0   b'\xab'             -- Binary encoding of the character '¼'
    1   b'\x62\x6f\x64\x6f' -- Binary encoding of the string 'bodo'
    2   b'\x4a\x2f\x31\x32' -- Binary encoding of the string 'J/12'
    ```
    Upon query
    ```sql
    SELECT TO_BINARY(b)
    FROM table1
    ```
    we will see a value error because all of the values in column b are not valid
    hex strings:
    - `'ABC'` is 3 characters, which is not an even number
    - `'ZETA'` contains non-hex characters `Z` and `T`
    - `'#fizz'` is 5 characters, which is not an even number and contains non-hex
    characters `#`, `i` and `z`


#### TRY_TO_BINARY
-  `TRY_TO_BINARY(COLUMN_EXPRESSION)`

    See `TO_BINARY`. The only difference is that `TRY_TO_BINARY` will return `NULL` upon
    encountering an invalid expression instead of raising an exception.


#### TO_CHAR
-  `#!sql TO_CHAR(COLUMN_EXPRESSION)`

    Casts the input to a string value. If the input is a boolean, it will be cast to `'true'` if it is `true` and `'false'` if it is `false`. If the input is `NULL`, the output will be `NULL`.

    _Example:_

    We are given `table1` with columns `a` and `b` and `c`
    ```python
    table1 = pd.DataFrame({
        'a': [1.1, 0, 2],
        'b': [True, False, True],
        'c': [None, 1, 0]
    })
    ```
    upon query
    ```sql
    SELECT
        TO_CHAR(a) AS a,
        TO_CHAR(b) AS b,
        TO_CHAR(c) AS c
    FROM table1;
    ```
    we will get the following output:
    ```
        a      b      c
    0  1.1   true   <NA>
    1    0  false      1
    2    2   true      0
    ```

#### TO_VARCHAR
-  `#!sql TO_VARCHAR(COLUMN_EXPRESSION)`

    Alias for `#!sql TO_CHAR`.

#### TO_DOUBLE
-  `#!sql TO_DOUBLE(COLUMN_EXPRESSION)`

    Converts a numeric or string expression to a double-precision floating-point number.
    For `NULL` input, the result is `NULL`.
    Fixed-point numbers are converted to floating point; the conversion cannot
    fail, but might result in loss of precision.
    Strings are converted as decimal or integer numbers. Scientific notation
    and special values (nan, inf, infinity) are accepted, case insensitive.

    _Example:_

    We are given `table1` with columns `a` and `b`
    ```python
    table1 = pd.DataFrame({
        'a': [1, 0, 2],
        'b': ['3.7', '-2.2e-1', 'nan'],
    })
    ```
    upon query
    ```sql
    SELECT
        TO_DOUBLE(a) AS a,
        TO_DOUBLE(b) AS b,
    FROM table1;
    ```
    we will get the following output:
    ```
           a      b
    0    1.0    3.7
    1    0.0  -0.22
    2    2.0   <NA>
    ```

#### TRY_TO_DOUBLE
-  `#!sql TRY_TO_DOUBLE(COLUMN_EXPRESSION)`

    This is similar to `#!sql TO_DOUBLE` except that it will return `NULL` instead of throwing an error invalid inputs.

#### TO_NUMBER
-   `#!sql TO_NUMBER(EXPR [, PRECICION [, SCALE]])`

    Converts an input expression to a fixed-point number with the specified precicion and scale.
    Precicon and scale must be constant integer literals if provided. Precicion must be between
    1 and 38. Scale must be between 0 and prec - 1.
    Precicion and scale default to 38 and 0 if not provided. For `NULL` input,
    the output is `NULL`.


#### TO_NUMERIC
-   `#!sql TO_NUMERIC(EXPR [, PRECICION [, SCALE]])`

    Equivalent to `#!sql TO_NUMBER(EXPR, PRECICION, SCALE)`.


#### TO_DECIMAL
-   `#!sql TO_DECIMAL(EXPR [, PRECICION [, SCALE]])`

    Equivalent to `#!sql TO_NUMBER(EXPR, PRECICION, SCALE)`.


#### TRY_TO_NUMBER
-   `#!sql TRY_TO_NUMBER(EXPR [, PRECICION [, SCALE]])`

    A special version of `#!sql TO_NUMBER` that performs
    the same operation (i.e. converts an input expression to a fixed-point
    number), but with error-handling support (i.e. if the conversion cannot be
    performed, it returns a `NULL` value instead of raising an error).


#### TRY_TO_NUMERIC
-   `#!sql TRY_TO_NUMERIC(EXPR [, PRECICION [, SCALE]])`

    Equivalent to `#!sql TRY_TO_NUMBER(EXPR, PRECICION, SCALE)`.


#### TRY_TO_DECIMAL
-   `#!sql TRY_TO_DECIMAL(EXPR [, PRECICION [, SCALE]])`

    Equivalent to `#!sql TRY_TO_NUMBER(EXPR, PRECICION, SCALE)`.


#### TO_DATE
-   `#!sql TO_DATE(EXPR)`

    Converts an input expression to a `DATE` type. The input can be one of
    the following:

    - `#!sql TO_DATE(timestamp_expr)` truncates the timestamp to its date value.
    - `#!sql TO_DATE(string_expr)` if the string is in date format (e.g. `"1999-01-01"`)
    then it is convrted to a corresponding date. If the string represents an integer
    (e.g. `"123456"`) then it is interpreted as the number of seconds/milliseconds/microseconds/nanoseconds
    since `1970-01-1`. Which unit it is interpreted as depends on the magnitude of the number,
    in accordance with [the semantics used by Snowflake](https://docs.snowflake.com/en/sql-reference/functions/to_date#usage-notes).
    - `#!sql TO_DATE(string_expr, format_expr)` uses the format string to specify how to parse the
    string expression as a date. Uses the format string rules [as specified by Snowflake](https://docs.snowflake.com/en/sql-reference/functions-conversion#label-date-time-format-conversion).
    - If the input is `NULL`, outputs `NULL`.

    Raises an error if the input expression does not match one of these formats.

#### TRY_TO_DATE
-   `#!sql TRY_TO_DATE(EXPR)`

    A special version of `#!sql TO_DATE` that performs
    the same operation but returns `NULL` instead of raising an error if
    something goes wrong during the conversion.

#### TO_TIME
-   `#!sql TO_TIME(EXPR)`

    Converts an input expression to a `TIME` type. The input can be one of
    the following:

    - `#!sql TO_TIME(timestamp_expr)` extracts the time component from a timestamp.
    - `#!sql TO_TIME(string_expr)` if the string is in date format (e.g. `"12:30:15"`)
    then it is convrted to a corresponding time.
    - `#!sql TO_TIME(string_expr, format_expr)` uses the format string to specify how to parse the
    string expression as a time. Uses the format string rules [as specified by Snowflake](https://docs.snowflake.com/en/sql-reference/functions-conversion#label-date-time-format-conversion).
    - If the input is `NULL`, outputs `NULL`

    Raises an error if the input expression does not match one of these formats.

#### TRY_TO_TIME
-   `#!sql TRY_TO_TIME(EXPR)`

    A special version of `#!sql TO_TIME` that performs
    the same operation but returns `NULL` instead of raising an error if
    something goes wrong during the conversion.

#### TO_TIMESTAMP
-   `#!sql TO_TIMESTAMP(EXPR)`

    Converts an input expression to a `TIMESTAMP` type without a timezone. The input can be one of
    the following:

    - `#!sql TO_TIMESTAMP(date_expr)` upcasts a `DATE` to a `TIMESTAMP`.
    - `#!sql TO_TIMESTAMP(integer)` creates a timestamp using the integer as the number of
    seconds/milliseconds/microseconds/nanoseconds since `1970-01-1`. Which unit it is interpreted
    as depends on the magnitude of the number, in accordance with [the semantics used by Snowflake](https://docs.snowflake.com/en/sql-reference/functions/to_date#usage-notes).
    - `#!sql TO_TIMESTAMP(integer, scale)` the same as the integer case except that the scale provided specifes which
    unit is used. THe scale can be an integer constant between 0 and 9, where 0 means seconds and 9 means nanoseconds.
    - `#!sql TO_TIMESTAMP(string_expr)` if the string is in timestamp format (e.g. `"1999-12-31 23:59:30"`)
    then it is convrted to a corresponding timestamp. If the string represents an integer
    (e.g. `"123456"`) then it uses the same rule as the corresponding input integer.
    - `#!sql TO_TIMESTAMP(string_expr, format_expr)` uses the format string to specify how to parse the
    string expression as a timestamp. Uses the format string rules [as specified by Snowflake](https://docs.snowflake.com/en/sql-reference/functions-conversion#label-date-time-format-conversion).
    - `#!sql TO_TIMESTAMP(timestamp_exr)` returns a timestamp expression representing the same moment in time,
    but changing the timezone if necessary to be timezone-naive.
    - If the input is `NULL`, outputs `NULL`

    Raises an error if the input expression does not match one of these formats.

#### TRY_TO_TIMESTAMP
-   `#!sql TRY_TO_TIMESTAMP(EXPR)`

    A special version of `#!sql TO_TIMESTAMP` that performs
    the same operation but returns `NULL` instead of raising an error if
    something goes wrong during the conversion.

#### TO_TIMESTAMP_NTZ
-   `#!sql TO_TIMESTAMP_NTZ(EXPR)`

    Equivalent to `#!sql TO_TIMESTAMP`.

#### TRY_TO_TIMESTAMP_NTZ
-   `#!sql TRY_TO_TIMESTAMP_NTZ(EXPR)`

    Equivalent to `#!sql TRY_TO_TIMESTAMP`.

#### TO_TIMESTAMP_LTZ
-   `#!sql TO_TIMESTAMP_LTZ(EXPR)`

    Equivalent to `#!sql TO_TIMESTAMP` except that it uses the local time zone.

#### TRY_TO_TIMESTAMP_LTZ
-   `#!sql TRY_TO_TIMESTAMP_NTZ(EXPR)`

    Equivalent to `#!sql TRY_TO_TIMESTAMP` except that it uses the local time zone.

#### TO_TIMESTAMP_TZ
-   `#!sql TO_TIMESTAMP_LTZ(EXPR)`

    Equivalent to `#!sql TO_TIMESTAMP` except that it uses the local time zone, or keeps
    the original timezone if the input is a timezone-aware timestamp.

#### TRY_TO_TIMESTAMP_TZ
-   `#!sql TRY_TO_TIMESTAMP_NTZ(EXPR)`

    Equivalent to `#!sql TRY_TO_TIMESTAMP` except that it uses the local time zone, or keeps
    the original timezone if the input is a timezone-aware timestamp.


### Table Functions
Bodo currently supports the following functions that produce tables:


#### FLATTEN
-   `#!sql FLATTEN([INPUT=>]expr[, PATH=>path_epxr][, OUTER=>outer_expr][, RECURSIVE=>recursive_expr][, MODE=>mode_epxr])`

    Takes in a column of semi-structured data and produces a table by
    "exploding" the data into multiple rows, producing the following
    columns:

    - `#!sql SEQ`: not currently supported by BodoSQL.
    - `#!sql KEY`: the individual values from the json data.
    - `#!sql PATH`: not currently supported by BodoSQL.
    - `#!sql INDEX`: the index within the array that the value came from.
    - `#!sql VALUE`: the individual values from the array or json data.
    - `#!sql THIS`: a copy of the input data.

    The function has the following named arguments:

    - `#!sql INPUT` (required): the expression of semi-structured data to flatten. Also allowed to be passed in as a positional argument without the `INPUT` keyword.
    - `#!sql PATH` (optional): a constant expression referencing how to access the semi-structured data to flatten from the input expression. BodoSQL currently only supports when this argument is omitted or is an empty string (indicating that the expression itself is the array to flatten).
    - `#!sql OUTER` (optional): a boolean indicating if a row should be generated even if the input data is empty. BodoSQL currently only supports when this argument is omitted or is false (which is the default).
    - `#!sql RECURSIVE` (optional): a boolean indicating if flattening should occur recursively, as opposed to just on the data referenced by `PATH`. BodoSQL currently only supports when this argument is omitted or is false (which is the default).
    - `#!sql MODE` (optional): a string literal that can be either `'OBJECT'`, `'ARRAY'` or `'BOTH'`, indicating what type of flattening rule should be done. BodoSQL currently only supports when this argument is omitted or is `'BOTH'` (which is the default).

    !!! Note: Snowflake supports the input being an array, JSON,
    or variant, and also allows several different input arguments
    to further control the behavior [(see here for more details)](https://docs.snowflake.com/en/sql-reference/functions/flatten). BodoSQL has more limited type support; it can handle
    arrays and JSON with values of the same type.


    Below is an example of a query using the `#!sql FLATTEN` function with the
    `#!sql LATERAL` keyword to explode an array column while also
    replicating another column.

    ```sql
    SELECT id, lat.index as idx, lat.value as val FROM table1, lateral flatten(tags) lat
    ```

    If the input data was as follows:

    | id | tags                      |
    |----|---------------------------|
    | 10 | ["A", "B"]                |
    | 16 | []                        |
    | 72 | ["C", "A", "B", "D", "C"] |

    Then the query would produce the following data:

    | id | idx | val |
    |----|-----|-----|
    | 10 | 0   | "A" |
    | 10 | 1   | "B" |
    | 72 | 0   | "C" |
    | 72 | 1   | "A" |
    | 72 | 2   | "B" |
    | 72 | 3   | "D" |
    | 72 | 4   | "C" |

    Below is an example of a query using the `#!sql FLATTEN` function with the
    `#!sql LATERAL` keyword to explode an JSON column while also
    replicating another column.

    ```sql
    SELECT id, lat.key as key, lat.value as val FROM table1, lateral flatten(attributes) lat
    ```

    If the input data was as follows:

    | id | attributes       |
    |----|------------------|
    | 42 | {"A": 0}         |
    | 50 | {}               |
    | 64 | {"B": 1, "C": 2} |

    Then the query would produce the following data:

    | id | key | value |
    |----|-----|-------|
    | 42 | "A" | 0     |
    | 64 | "B" | 1     |
    | 64 | "C" | 2     |


#### SPLIT_TO_TABLE
-   `#!sql SPLIT_TO_TABLE(str, delim)`

    Takes in a string column and a delimeter and produces a table by
    "exploding" the string into multiple rows based on the delimeter,
    producing the following columns:

    - `#!sql SEQ`: not currently supported by BodoSQL.
    - `#!sql INDEX`: which index in the splitted string did the current seciton come from.
    - `#!sql VALUE`: the current section of the splitted string.

    !!! Note: Currently, BodoSQL supports this function as an alias
    for `#!sql FLATTEN(SPLIT(str, delim))`.

    Below is an example of a query using the `#!sql SPLIT_TO_TABLE` function with the
    `#!sql LATERAL` keyword to explode an string column while also
    replicating another column.

    ```sql
    SELECT id, lat.index as idx, lat.value as val FROM table1, lateral split_to_table(colors, ' ') lat
    ```

    If the input data was as follows:

    | id | colors              |
    |----|---------------------|
    | 50 | "red orange yellow" |
    | 75 | "green blue"        |

    Then the query would produce the following data:

    | id | idx | val      |
    |----|-----|----------|
    | 50 | 0   | "red"    |
    | 50 | 1   | "orange" |
    | 50 | 2   | "yellow" |
    | 75 | 0   | "green"  |
    | 75 | 1   | "blue"   |


###   Context Functions (Session Object)

#### CURRENT_DATABASE
-   `#!sql CURRENT_DATABASE()`

    Returns the name of the database in use for the current session.


## Supported DataFrame Data Types

BodoSQL uses Pandas DataFrames to represent SQL tables in memory and
converts SQL types to corresponding Python types which are used by Bodo.
Below is a table mapping SQL types used in BodoSQL to their respective
Python types and Bodo data types.


<center>

|SQL Type(s)           |Equivalent Python Type  |Bodo Data Type       |
|----------------------|------------------------|---------------------|
|`TINYINT`             |`np.int8`               |`bodo.int8`          |
|`SMALLINT`            |`np.int16`              |`bodo.int16`         |
|`INT`                 |`np.int32`              |`bodo.int32`         |
|`BIGINT`              |`np.int64`              |`bodo.int64`         |
|`FLOAT`               |`np.float32`            |`bodo.float32`       |
|`DECIMAL`, `DOUBLE`   |`np.float64`            |`bodo.float64`       |
|`VARCHAR`, `CHAR`     |`str`                   |`bodo.string_type`   |
|`TIMESTAMP`, `DATE`   |`np.datetime64[ns]`     |`bodo.datetime64ns`  |
|`INTERVAL(day-time)`  |`np.timedelta64[ns]`    |`bodo.timedelta64ns` |
|`BOOLEAN`             |`np.bool_`              |`bodo.bool_`         |

</center>

BodoSQL can also process DataFrames that contain Categorical or Date
columns. However, Bodo will convert these columns to one of the
supported types, which incurs a performance cost. We recommend
restricting your DataFrames to the directly supported types when
possible.

### Nullable and Unsigned Types

Although SQL does not explicitly support unsigned types, by default,
BodoSQL maintains the exact types of the existing DataFrames registered
in a [BodoSQLContext], including unsigned and non-nullable
type behavior. If an operation has the possibility of creating null
values or requires casting data, BodoSQL will convert the input of that
operation to a nullable, signed version of the type.

## Supported Literals

BodoSQL supports the following literal types:

-   `#!sql boolean_literal`
-   `#!sql datetime_literal`
-   `#!sql float_literal`
-   `#!sql integer_literal`
-   `#!sql interval_literal`
-   `#!sql string_literal`

### Boolean Literal {#boolean_literal}

**Syntax**:

```sql
TRUE | FALSE
```

Boolean literals are case-insensitive.

### Datetime Literal {#datetime_literal}

**Syntax**:

```sql
DATE 'yyyy-mm-dd' |
TIMESTAMP 'yyyy-mm-dd' |
TIMESTAMP 'yyyy-mm-dd HH:mm:ss'
```
### Float Literal {#float_literal}

**Syntax**:

```sql
[ + | - ] { digit [ ... ] . [ digit [ ... ] ] | . digit [ ... ] }
```

where digit is any numeral from 0 to 9

### Integer Literal {#integer_literal}

**Syntax**:

```sql
[ + | - ] digit [ ... ]
```

where digit is any numeral from 0 to 9

### Interval Literal {#interval_literal}

**Syntax**:

```sql
INTERVAL integer_literal interval_type
```

Where integer_literal is a valid integer literal and interval type is
one of:

```sql
DAY[S] | HOUR[S] | MINUTE[S] | SECOND[S]
```

In addition, we also have limited support for `#!sql YEAR[S]` and `#!sql MONTH[S]`.
These literals cannot be stored in columns and currently are only
supported for operations involving add and sub.

### String Literal {#string_literal}

**Syntax**:

```sql
'char [ ... ]'
```

Where char is a character literal in a Python string.


## BodoSQL Caching & Parameterized Queries {#bodosql_named_params}

BodoSQL can reuse Bodo caching to avoid recompilation when used inside a
JIT function. BodoSQL caching works the same as Bodo, so for example:

```py
@bodo.jit(cache=True)
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"table1": df1})
    df2 = bc.sql("SELECT A FROM table1 WHERE B > 4")
    print(df2.A.sum())
```

This will avoid recompilation so long as the DataFrame scheme stored in
`filename` has the same schema and the code does not change.

To enable caching for queries with scalar parameters that you may want
to adjust between runs, we introduce a feature called parameterized
queries. In a parameterized query, the SQL query replaces a
constant/scalar value with a variable, which we call a named parameter.
In addition, the query is passed a dictionary of parameters which maps
each name to a corresponding Python variable.

For example, if in the above SQL query we wanted to replace 4 with other
integers, we could rewrite our query as:

```py
bc.sql("SELECT A FROM table1 WHERE B @var", {"var": python_var})
```

Now anywhere that `@var` is used, the value of python_var at runtime
will be used instead. This can be used in caching, because python_var
can be provided as an argument to the JIT function itself, thus enabling
changing the filter without recompiling. The full example looks like
this:

```py
@bodo.jit(cache=True)
def f(filename, python_var):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"table1": df1})
    df2 = bc.sql("SELECT A FROM table1 WHERE B @var", {"var": python_var})
    print(df2.A.sum())
```

Named parameters cannot be used in places that require a constant value
to generate the correct implementation (e.g. TimeUnit in EXTRACT).


## IO Handling

BodoSQL is great for compute based SQL queries, but you cannot yet access external storage directly from SQL. Instead, you can load and store data using Bodo and various Python APIs. Here we explain a couple common methods for loading data.

### Pandas IO in JIT function with SQL Query

The most common way to load data is to first use Pandas APIs to load a DataFrame inside a JIT function and then to use that DataFrame inside a BodoSQLContext.

```py
def f(f1, f2):
    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)
    bc = bodosql.BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
        }
    )
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")
```


### Pandas IO in a JIT Function Separate from Query

The previous approach works well for most individual queries. However, when running several queries on the same dataset, it should ideally be loaded once for all queries. To do this, you can structure your JIT code to contain a single load function at the beginning. For example:

```py

@bodo.jit
def load_data(f1, f2):
    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)
    return df1, df2

def q1(df1, df2):
    bc = bodosql.BodoSQLContext(
        {
            "t1": df1,
            "t2": df2,
        }
    )
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")

...

@bodo.jit
def run_queries(f1, f2):
    df1, df2 = load_data(f1, f2)
    print(q1(df1, df2))
    print(q2(df2))
    print(q3(df1))
    ...

run_queries(f1, f2)
```

This approach prevents certain optimizations, such as filter pushdown. However, the assumption here is that you will use the entire DataFrame across the various benchmarks, so no optimization is useful by itself. In addition, any optimizations that can apply to all queries can be done explicitly inside `load_data`. For example, if all queries are operate on a single day's data with `df1`, you can write that filter in `load_data` to limit IO and filter pushdown will be performed.

```py

@bodo.jit
def load_data(f1, f2, target_date):
    df1 = pd.read_parquet(f1)
    # Applying this filter limits how much data is loaded.
    df1 = df1[df1.date_val == target_date]
    df2 = pd.read_parquet(f2)
    return df1, df2

@bodo.jit
def run_queries(f1, f2, target_date):
    df1, df2 = load_data(f1, f2, target_date)
    ...

run_queries(f1, f2, target_date)
```

## BodoSQLContext API

The `BodoSQLContext` API is the primary interface for executing SQL queries. It performs two roles:

  1. Registering data and connection information to load tables of interest.
  2. Forwarding SQL queries to the BodoSQL engine for compilation and execution. This is done via the
     `bc.sql(query)` method, where `bc` is a `BodoSQLContext` object.



A `BodoSQLContext` can be defined in regular Python and passed as an argument to JIT functions or can be
defined directly inside JIT functions. We recommend defining and modifying a `BodoSQLContext` in regular
Python whenever possible.

For example:

```py
bc = bodosql.BodoSQLContext(
    {
        "t1": bodosql.TablePath("my_file_path.pq", "parquet"),
    },
    catalog=bodosql.SnowflakeCatalog(
        username,
        password,
        account_name,
        warehouse_name,
        database name,
    )
)

@bodo.jit
def f(bc):
    return bc.sql("select t1.A, t2.B from t1, catalogSchema.t2 where t1.C > 5 and t1.D = catalogSchema.t2.D")
```

### API Reference

- ++bodosql.%%BodoSQLContext%%(tables: Optional[Dict[str, Union[pandas.DataFrame|TablePath]]] = None, catalog: Optional[DatabaseCatalog] = None)++
<br><br>

    Defines a `BodoSQLContext` with the given local tables and catalog.

    ***Arguments***

    - `tables`: A dictionary that maps a name used in a SQL query to a `DataFrame` or `TablePath` object.

    - `catalog`: A `DatabaseCatalog` used to load tables from a remote database (e.g. Snowflake).


- ++bodosql.BodoSQLContext.%%sql%%(self, query: str, params_dict: Optional[Dict[str, Any] = None)++
<br><br>

    Executes a SQL query using the tables registered in this `BodoSQLContext`. This function should
    be used inside a `@bodo.jit` function.

    ***Arguments***

    - `query`: The SQL query to execute. This function generates code that is compiled so the `query` argument is required
    to be a compile time constant.

   -  `params_dict`: A dictionary that maps a SQL usable name to Python variables. For more information please
    refer to [the BodoSQL named parameters section][bodosql_named_params].

    ***Returns***

    A `DataFrame` that results from executing the query.


- ++bodosql.BodoSQLContext.%%add_or_replace_view%%(self, name: str, table: Union[pandas.DataFrame, TablePath])++
<br><br>

    Create a new `BodoSQLContext` from an existing `BodoSQLContext` by adding or replacing a table.

    ***Arguments***

    - `name`: The name of the table to add. If the name already exists references to that table
    are removed from the new context.

    - `table`: The table object to add. `table` must be a `DataFrame` or `TablePath` object.

    ***Returns***

    A new `BodoSQLContext` that retains the tables and catalogs from the old `BodoSQLContext` and inserts the new table specified.

    !!! note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.add_or_replace_view("t1", table)`



- ++bodosql.BodoSQLContext.%%remove_view%%(self, name: str)++
<br><br>

    Creates a new `BodoSQLContext` from an existing context by removing the table with the
    given name. If the name does not exist, a `BodoError` is thrown.

    ***Arguments***

    - `name`: The name of the table to remove.

    ***Returns***

    A new `BodoSQLContext` that retains the tables and catalogs from the old `BodoSQLContext` minus the table specified.

    !!! note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.remove_view("t1")`


- ++bodosql.BodoSQLContext.%%add_or_replace_catalog%%(self, catalog: DatabaseCatalog)++
<br><br>

    Create a new `BodoSQLContext` from an existing context by replacing the `BodoSQLContext` object's `DatabaseCatalog` with
    a new catalog.

    ***Arguments***

    - `catalog`: The catalog to insert.

    ***Returns***

    A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but replaces the old catalog with the new catalog specified.

    !!! note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.add_or_replace_catalog(catalog)`


- ++bodosql.BodoSQLContext.%%remove_catalog%%(self)++
<br><br>

    Create a new `BodoSQLContext` from an existing context by removing its `DatabaseCatalog`.

    ***Returns***

    A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but removes the old catalog.

    !!!note
        This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
        e.g. `bc = bc.remove_catalog()`


## TablePath API

The `TablePath` API is a general purpose IO interface to specify IO sources. This API is meant
as an alternative to natively loading tables in Python inside JIT functions.
The `TablePath` API stores the user-defined data location and the storage type to load a table of interest.
For example, here is some sample code that loads two DataFrames from parquet using the `TablePath` API.

```py
bc = bodosql.BodoSQLContext(
    {
        "t1": bodosql.TablePath("my_file_path1.pq", "parquet"),
        "t2": bodosql.TablePath("my_file_path2.pq", "parquet"),
    }
)

@bodo.jit
def f(bc):
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")
```

Here, the `TablePath` constructor doesn't load any data. Instead, a `BodoSQLContext` internally generates code to load the tables of interest after parsing the SQL query. Note that a `BodoSQLContext` loads all used tables from I/O *on every query*, which means that if users would like to perform multiple queries on the same data, they should consider loading the DataFrames once in a separate JIT function.

### API Reference

- ++bodosql.%%TablePath%%(file_path: str, file_type: str, *, conn_str: Optional[str] = None, reorder_io: Optional[bool] = None)++
<br><br>

    Specifies how a DataFrame should be loaded from IO by a BodoSQL query. This
    can only load data when used with a `BodoSQLContext` constructor.

    ***Arguments***

    - `file_path`: Path to IO file or name of the table for SQL. This must constant at compile time if used inside JIT.

    - `file_type`: Type of file to load as a string. Supported values are ``"parquet"`` and ``"sql"``. This must constant at compile time if used inside JIT.

    - `conn_str`: Connection string used to connect to a SQL DataBase, equivalent to the conn argument to `pandas.read_sql`. This must be constant at compile time if used inside JIT and must be None if not loading from a SQL DataBase.

   - `reorder_io`: Boolean flag determining when to load IO. If `False`, all used tables are loaded before executing any of the query. If `True`, tables are loaded just before first use inside the query, which often results in decreased
    peak memory usage as each table is partially processed before loading the next table. The default value, `None`, behaves like `True`, but this may change in the future. This must be constant at compile time if used inside JIT.


## Database Catalogs

Database Catalogs are configuration objects that grant BodoSQL access to load tables from a remote database.
For example, when a user wants to load data from Snowflake, a user will create a `SnowflakeCatalog` to grant
BodoSQL access to their Snowflake account and load the tables of interest.

A database catalog can be registered during the construction of the `BodoSQLContext` by passing it in as a parameter, or can be manually set using the
`BodoSQLContext.add_or_replace_catalog` API. Currently, a `BodoSQLContext` can support at most one database catalog.

When using a catalog in a `BodoSQLContext` we strongly recommend creating the `BodoSQLContext` once in regular Python and then
passing the `BodoSQLContext` as an argument to JIT functions. There is no benefit to creating the
`BodoSQLContext` in JIT and this could increase compilation time.

```py
catalog = bodosql.SnowflakeCatalog(
    username,
    password,
    account_name,
    "DEMO_WH", # warehouse name
    "SNOWFLAKE_SAMPLE_DATA", # database name
)
bc = bodosql.BodoSQLContext({"local_table1": df1}, catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT r_name, local_id FROM TPCH_SF1.REGION, local_table1 WHERE R_REGIONKEY = local_table1.region_key ORDER BY r_name")

run_query(bc)
```

Database catalogs can be used alongside local, in-memory `DataFrame` or `TablePath` tables. If a table is
specified without a schema then BodoSQL resolves the table in the following order:

1. Default Catalog Schema
2. Local (in-memory) DataFrames / TablePath names

An error is raised if the table cannot be resolved after searching through both of these data sources.

This ordering indicates that in the event of a name conflict between a table in the database catalog and a local table, the table in the database catalog is used.

If a user wants to use the local table instead, the user can explicitly specify the table with the local schema `__bodolocal__`.

For example:

```SQL
SELECT A from __bodolocal__.table1
```

Currently, BodoSQL supports catalogs Snowflake, but support for other data storage systems will be added in future releases.

### SnowflakeCatalog

The Snowflake Catalog offers an interface for users to connect their Snowflake accounts to use with BodoSQL.
With a Snowflake Catalog, users only have to specify their Snowflake connection once, and can access any tables of interest in their Snowflake account. Currently, the Snowflake Catalog is defined to
use a single `DATABASE` (e.g. `USE DATABASE`)  at a time, as shown below.

```py

catalog = bodosql.SnowflakeCatalog(
    username,
    password,
    account_name,
    "DEMO_WH", # warehouse name
    "SNOWFLAKE_SAMPLE_DATA", # database name
)
bc = bodosql.BodoSQLContext(catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT r_name FROM TPCH_SF1.REGION ORDER BY r_name")

run_query(bc)
```

BodoSQL does not currently support Snowflake syntax for specifying defaults
and session parameters (e.g. `USING SCHEMA <NAME>`). Instead users can pass
any session parameters through the optional `connection_params` argument, which
accepts a `Dict[str, str]` for each session parameter. For example, users can provide
a default schema to simplify the previous example.

```py

catalog = bodosql.SnowflakeCatalog(
    username,
    password,
    account,
    "DEMO_WH", # warehouse name
    "SNOWFLAKE_SAMPLE_DATA", # database name
    connection_params={"schema": "TPCH_SF1"}
)
bc = bodosql.BodoSQLContext(catalog=catalog)

@bodo.jit
def run_query(bc):
    return bc.sql("SELECT r_name FROM REGION ORDER BY r_name")

run_query(bc)
```

Internally, Bodo uses the following connections to Snowflake:

1. A JDBC connection to lazily fetch metadata.
2. The Snowflake-Python-Connector's distributed fetch API to load batches of arrow data.

#### API Reference

- ++bodosql.%%SnowflakeCatalog%%(username: str, password: str, account: str, warehouse: str, database: str, connection_params: Optional[Dict[str, str]] = None)++
<br><br>

    Constructor for `SnowflakeCatalog`. Allows users to execute queries on tables stored in Snowflake when the `SnowflakeCatalog` object is registered with a `BodoSQLContext`.

    ***Arguments***

    - `username`: Snowflake account username.

    - `password`: Snowflake account password.

    - `account`: Snowflake account name.

    - `warehouse`: Snowflake warehouse to use when loading data.

    - `database`: Name of Snowflake database to load data from. The Snowflake
        Catalog is currently restricted to using a single Snowflake `database`.

    - `connection_params`: A dictionary of Snowflake session parameters.


#### Supported Query Types

The `SnowflakeCatalog` currently supports the following types of SQL queries:

  * `#!sql SELECT`
  * `#!sql INSERT INTO`
  * `#!sql DELETE`


## Performance Considerations

### Snowflake Views

Users may define views within their Snowflake account to enable greater query reuse.
Views may constitute performance bottlenecks because if a view is evaluated in Snowflake
Bodo will need to wait for the result before it can fetch data and may have less access
to optimizations.

To improve performance in these circumstances Bodo will attempt to expand any views into
the body of the query to allow Bodo to operate on the underlying tables. When this occurs
users should face no performance penalty for using views in their queries. However there are
a few situations in which this is not possible, namely

  * The Snowflake User passed to Bodo does not have permissions to determine the view definition.
  * The Snowflake User passed to Bodo does not have permissions to
    read all of the underlying tables.
  * The view is a materalized or secure view.

If for any reason Bodo is unable to expand the view, then the query will execute treating
the view as a table and delegate it to Snowflake.
