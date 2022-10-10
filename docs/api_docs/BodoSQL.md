Bodo SQL {#bodosql}
========

Bodo SQL provides high performance and scalable SQL query execution
using Bodo's HPC capabilities and optimizations. It also provides
native Python/SQL integration as well as SQL to Pandas conversion for
the first time. BodoSQL is in early stages and its capabilities are
expanding rapidly.

## Getting Started

### Installation

Bodo SQL is currently in Beta. Install it using:

```shell
conda install bodosql -c bodo.ai -c conda-forge
```

### Using Bodo SQL

The example below demonstrates using Bodo SQL in Python programs. It
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

To run the example, save it in a file called `example.py` and run it using mpiexec, e.g.:

```console
mpiexec -n 8 python example.py
```

## Aliasing

In all but the most trivial cases, Bodo SQL generates internal names to
avoid conflicts in the intermediate dataframes. By default, Bodo SQL
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
     BodoSQL supports using aliases generated in `SELECT` inside
    `GROUP BY` and `HAVING` in the same query, but you cannot do so with
    `WHERE`.

## Supported Operations

We currently support the following SQL query statements and clauses with
Bodo SQL, and are continuously adding support towards completeness. Note
that Bodo SQL ignores casing of keywords, and column and table names,
except for the final output column name. Therefore,
`select a from table1` is treated the same as `SELECT A FROM Table1`,
except for the names of the final output columns (`a` vs `A`).

### SELECT

The `SELECT` statement is used to select data in the form of
columns. The data returned from Bodo SQL is stored in a dataframe.

```sql
SELECT <COLUMN_NAMESFROM <TABLE_NAME>
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

The `SELECT DISTINCT` statement is used to return only distinct
(different) values:

```sql
SELECT DISTINCT <COLUMN_NAMESFROM <TABLE_NAME>
```

`DISTINCT` can be used in a SELECT statement or inside an
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

The `WHERE` clause on columns can be used to filter records that
satisfy specific conditions:

```sql
SELECT <COLUMN_NAMESFROM <TABLE_NAMEWHERE <CONDITION>
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

The `ORDER BY` keyword sorts the resulting dataframe in ascending
or descending order, with NULL values either at the start or end
of the column. By default, it sorts the records in ascending order
with null values at the end. For descending order and nulls at the
front, the `DESC` and `NULLS FIRST` keywords can be used:

```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
ORDER BY <ORDERED_COLUMN_NAMES[ASC|DESC] [NULLS FIRST|LAST]
```

For Example:
```sql
SELECT A, B FROM table1 ORDER BY B, A DESC NULLS FIRST
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

Bodo SQL supports the `LIMIT` keyword to select a limited number
of rows. This keyword can optionally include an offset:

```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <CONDITION>
LIMIT <LIMIT_NUMBEROFFSET <OFFSET_NUMBER>
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

The `IN` determines if a value can be chosen a list of options.
Currently we support lists of literals or columns with matching
types:
```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <COLUMN_NAMEIN (<val1>, <val2>, ... <valN>)
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

The `BETWEEN` operator selects values within a given range. The
values can be numbers, text, or datetimes. The `BETWEEN` operator
is inclusive: begin and end values are included:
```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <COLUMN_NAMEBETWEEN <VALUE1AND <VALUE2>
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

THE `CAST` operator converts an input from one type to another. In
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
    CAST correctness can often not be determined at compile time.
    Users are responsible for ensuring that conversion is possible
    (e.g. `CAST(str_col as INTEGER)`).

### JOIN

A `JOIN` clause is used to combine rows from two or more tables,
based on a related column between them:
```sql
SELECT <COLUMN_NAMES>
  FROM <LEFT_TABLE_NAME>
  <JOIN_TYPE<RIGHT_TABLE_NAME>
  ON <LEFT_TABLE_COLUMN_NAME= <RIGHT_TABLE_COLUMN_NAME>
```
For example:
```sql
SELECT table1.A, table1.B FROM table1 JOIN table2 on table1.A = table2.C
```
Here are the different types of the joins in SQL:

-   `(INNER) JOIN`: returns records that have matching values in
both tables
-   `LEFT (OUTER) JOIN`: returns all records from the left table,
and the matched records from the right table
-   `RIGHT (OUTER) JOIN`: returns all records from the right
table, and the matched records from the left table
-   `FULL (OUTER) JOIN`: returns all records when there is a match
in either left or right table

Bodo SQL currently supports inner join on all conditions, but all
outer joins are only supported on an equality between columns.

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

### UNION

The UNION operator is used to combine the result-set of two SELECT
statements:
```sql
SELECT <COLUMN_NAMESFROM <TABLE1>
UNION
SELECT <COLUMN_NAMESFROM <TABLE2>
```
Each SELECT statement within the UNION clause must have the same
number of columns. The columns must also have similar data types.
The output of the UNION is the set of rows which are present in
either of the input SELECT statements.

The UNION operator selects only the distinct values from the
inputs by default. To allow duplicate values, use UNION ALL:

```sql
SELECT <COLUMN_NAMESFROM <TABLE1>
UNION ALL
SELECT <COLUMN_NAMESFROM <TABLE2>
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

The INTERSECT operator is used to calculate the intersection of
two SELECT statements:

```sql
SELECT <COLUMN_NAMESFROM <TABLE1>
INTERSECT
SELECT <COLUMN_NAMESFROM <TABLE2>
```

Each SELECT statement within the INTERSECT clause must have the
same number of columns. The columns must also have similar data
types. The output of the INTERSECT is the set of rows which are
present in both of the input SELECT statements. The INTERSECT
operator selects only the distinct values from the inputs.

### GROUP BY

The `GROUP BY` statement groups rows that have the same values
into summary rows, like "find the number of customers in each
country". The `GROUP BY` statement is often used with aggregate
functions to group the result-set by one or more columns:
```sql
SELECT <COLUMN_NAMES>
FROM <TABLE_NAME>
WHERE <CONDITION>
GROUP BY <GROUP_EXPRESION>
ORDER BY <COLUMN_NAMES>
```

For example:
```sql
SELECT MAX(A) FROM table1 GROUP BY B
```
`GROUP BY` statements also referring to columns by alias or
column number:
```sql
SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val
SELECT MAX(A), B FROM table1 GROUP BY 2
```

BodoSQL supports several subclauses that enable grouping by multiple different
sets of columns in the same `SELECT` statment. `GROUPING SETS` is the first. It is
equivalent to performing a group by for each specified set (seting each column not
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

`CUBE` is equivalent to grouping by all possible permutations of the specified set.
For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY CUBE(B, C)
```

Is equivalent to

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS ((B, C), (B), (C), ())
```

`ROLLUP` is equivalent to grouping by n + 1 grouping sets, where each set is constructed by dropping the rightmost element from the previous set, until no elements remain in the grouping set. For example:

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY ROLLUP(B, C, D)
```

Is equivalent to

```sql
SELECT MAX(A), B, C FROM table1 GROUP BY GROUPING SETS ((B, C, D), (B, C), (B), ())
```

`CUBE` and `ROLLUP` can be nested into a `GROUPING SETS` clause. For example:

```sql
SELECT MAX(A), B, C GROUP BY GROUPING SETS (ROLLUP(B, C, D), CUBE(B, C), (A))
```

Which is equivalent to

```sql
SELECT MAX(A), B, C GROUP BY GROUPING SETS ((B, C, D), (B, C), (B), (), (B, C), (B), (C), (), (A))
```

### HAVING

The `HAVING` clause is used for filtering with `GROUP BY`.
`HAVING` applies the filter after generating the groups, whereas
`WHERE` applies the filter before generating any groups:
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
`HAVING` statements also referring to columns by aliases used in
the `GROUP BY`:
```sql
SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val HAVING val 5
```

### QUALIFY

`QUALIFY` is similar to `HAVING`, except it applies filters after computing the results of at least one window function. `QUALIFY` is used after using `WHERE` and `HAVING`.

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

The `CASE` statement goes through conditions and returns a value
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
excluded, in which case, the CASE statement will return null if
none of the conditions are met. For example:
```sql
SELECT (CASE WHEN A < 0 THEN 0 END) as mycol FROM table1
```
is equivalent to:
```sql
SELECT (CASE WHEN A < 0 THEN 0 ELSE NULL END) as mycol FROM table1
```

### LIKE

The `LIKE` clause is used to filter the strings in a column to
those that match a pattern:
```sql
SELECT column_name(s) FROM table_name WHERE column LIKE pattern
```
In the pattern we support the wildcards `%` and `_`. For example:
```sql
SELECT A FROM table1 WHERE B LIKE '%py'
```

### GREATEST

The `GREATEST` clause is used to return the largest value from a
list of columns:
```sql
SELECT GREATEST(col1, col2, ..., colN) FROM table_name
```
For example:
```sql
SELECT GREATEST(A, B, C) FROM table1
```

### LEAST

The `LEAST` clause is used to return the smallest value from a
list of columns:
```sql
SELECT LEAST(col1, col2, ..., colN) FROM table_name
```
For example:
```sql
SELECT LEAST(A, B, C) FROM table1
```

### PIVOT

The `PIVOT` clause is used to transpose specific data rows in one
or more columns into a set of columns in a new DataFrame:
```sql
SELECT col1, ..., colN FROM table_name PIVOT (
    AGG_FUNC_1(colName or pivotVar) AS alias1, ...,  AGG_FUNC_N(colName or pivotVar) as aliasN
    FOR pivotVar IN (ROW_VALUE_1 as row_alias_1, ..., ROW_VALUE_N as row_alias_N)
)
```
`PIVOT` produces a new column for each pair of pivotVar and
aggregation functions.

For example:
```sql
SELECT single_sum_a, single_avg_c, triple_sum_a, triple_avg_c FROM table1 PIVOT (
    SUM(A) AS sum_a, AVG(C) AS avg_c
    FOR A IN (1 as single, 3 as triple)
)
```
Here `single_sum_a` will contain sum(A) where `A = 1`,
single_avg_c will contain AVG(C) where `A = 1` etc.

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

The `WITH` clause can be used to name subqueries:
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
SELECT <COLUMN_NAMEAS <ALIAS>
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
-   Bodo SQL currently supports the following arithmetic
    operators:

    -   `+` (addition)
    -   `-` (subtraction)
    -   `*` (multiplication)
    -   `/` (true division)
    -   `%` (modulo)

#### Comparison
-   Bodo SQL currently supports the following comparison
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
-   Bodo SQL currently supports the following logical operators:

    -   `AND`
    -   `OR`
    -   `NOT`

#### String
-   Bodo SQL currently supports the following string operators:

    -   `||` (string concatenation)

###  Numeric Functions

Except where otherwise specified, the inputs to each of these
functions can be any numeric type, column or scalar. Here is an
example using MOD:

```sql
SELECT MOD(12.2, A) FROM table1
```

Bodo SQL Currently supports the following Numeric Functions:

#### ABS
-   `ABS(n)`

    Returns the absolute value of n

#### COS
-   `COS(n)`

    Calculates the Cosine of n

#### SIN
-   `SIN(n)`

    Calculates the Sine of n

#### TAN
-   `TAN(n)`

    Calculates the Tangent of n

#### ACOS
-   `ACOS(n)`

    Calculates the Arccosine of n

#### ASIN
-   `ASIN(n)`

    Calculates the Arcsine of n

#### ATAN
-   `ATAN(n)`

    Calculates the Arctangent of n

#### ATAN2
-   `ATAN2(A, B)`

    Calculates the Arctangent of `A` divided by `B`

#### COTAN
-   `COTAN(X)`

    Calculates the Cotangent of `X`

#### CEIL
-   `CEIL(X)`

    Converts X to an integer, rounding towards positive
    infinity

#### CEILING
-   `CEILING(X)`

    Equivalent to CEIL

#### FLOOR
-   `FLOOR(X)`

    Converts X to an integer, rounding towards negative infinity

#### DEGREES
-   `DEGREES(X)`

    Converts a value in radians to the corresponding value in
    degrees

#### RADIANS
-   `RADIANS(X)`

    Converts a value in radians to the corresponding value in
    degrees

#### LOG10
-   `LOG10(X)`

    Computes Log base 10 of x. Returns NaN for negative inputs,
    and -inf for 0 inputs.

#### LOG
-   `LOG(X)`

    Equivalent to LOG10(x)

#### LOG10
-   `LOG10(X)`

    Computes Log base 2 of x. Returns NaN for negative inputs,
    and -inf for 0 inputs.

#### LN
-   `LN(X)`

    Computes the natural log of x. Returns NaN for negative
    inputs, and -inf for 0 inputs.

#### MOD
-   `MOD(A,B)`

    Computes A modulo B.

#### CONV
-   `CONV(X, current_base, new_base)`

    `CONV` takes a string representation of an integer value,
    it's current_base, and the base to convert that argument
    to. `CONV` returns a new string, that represents the value in
    the new base. `CONV` is only supported for converting to/from
    base 2, 8, 10, and 16.

    For example:

    ```sql
    CONV('10', 10, 2) =='1010'
    CONV('10', 2, 10) =='2'
    CONV('FA', 16, 10) =='250'
    ```
#### SQRT
-   `SQRT(X)`

    Computes the square root of x. Returns NaN for negative
    inputs, and -inf for 0 inputs.

#### PI
-   `PI()`

    Returns the value of PI

#### POW, POWER
-   `POW(A, B), POWER(A, B)`

    Returns A to the power of B. Returns NaN if A is negative,
    and B is a float. POW(0,0) is 1

#### EXP
-   `EXP(X)`

    Returns e to the power of X

#### SIGN
-   `SIGN(X)`

    Returns 1 if X 0, -1 if X < 0, and 0 if X = 0

#### ROUND
-   `ROUND(X, num_decimal_places)`

    Rounds X to the specified number of decimal places

#### TRUNCATE
-   `TRUNCATE(X, num_decimal_places)`

    Equivalent to `ROUND(X, num_decimal_places)`


#### BITAND
-   `BITAND(A, B)`

    Returns the bitwise-and of its inputs.


#### BITOR
-   `BITOR(A, B)`

    Returns the bitwise-or of its inputs.


#### BITXOR
-   `BITOR(A, B)`

    Returns the bitwise-xor of its inputs.


#### BITNOT
-   `BITNOT(A)`

    Returns the bitwise-negation of its input.



#### BITSHIFTLEFT
-   `BITSHIFTLEFT(A, B)`

    Returns the bitwise-leftshift of its inputs.
    Note: the output is always of type int64.
    Undefined behavior when B is negative or
    too large.


#### BITSHIFTRIGHT
-   `BITSHIFTRIHGT(A, B)`

    Returns the bitwise-rightshift of its inputs.
    Undefined behavior when B is negative or
    too large.


#### GETBIT
-   `GETBIT(A, B)`

    Returns the bit of A corresponding to location B,
    where 0 is the rightmost bit. Undefined behavior when
    B is negative or too large.


#### BOOLAND
-   `BOOLAND(A, B)`

    Returns true when `A` and `B` are both non-null non-zero.
    Returns false when one of the arguments is zero and the
    other is either zero or `NULL`. Returns `NULL` otherwise.


#### BOOLOR
-   `BOOLOR(A, B)`

    Returns true if either `A` or `B` is non-null and non-zero.
    Returns false if both `A` and `B` are zero. Returns `NULL` otherwise.


#### BOOLXOR
-   `BOOLXOR(A, B)`

    Returns true if one of `A` and `B` is zero and the other is non-zero.
    Returns false if `A` and `B` are both zero or both non-zero. Returns
    `NULL` if either `A` or `B` is NULL.


#### BOOLNOT
-   `BOOLNOT(A)`

    Returns true if `A` is zero. Returns false if `A` is non-zero. Returns
    `NULL` if `A` is `NULL`.


#### REGR_VALX
-   `REGR_VALX(Y, X)`

    Returns `NULL` if either input is `NULL`, otherwise `X`


#### REGR_VALY
-   `REGR_VALY(Y, X)`

    Returns `NULL` if either input is `NULL`, otherwise `Y`

### Aggregation Functions

Bodo SQL Currently supports the following Aggregation Functions on
all types:


#### COUNT
-   `COUNT`

    Count the number of elements in a column or group.


#### ANY_VALUE
-   `ANY_VALUE`

    Select an arbitrary value. Note: currently BodoSQL always selects the first value,
    but this is subject to change at any time.


In addition, Bodo SQL also supports the following functions on
numeric types


#### AVG
-   `AVG`

    Compute the mean for a column.

#### MAX
-   `MAX`

    Compute the max value for a column.

#### MIN
-   `MIN`

    Compute the min value for a column.

#### STDDEV
-   `STDDEV`

    Compute the standard deviation for a column with N - 1
    degrees of freedom.

#### STDDEV_SAMP
-   `STDDEV_SAMP`

    Compute the standard deviation for a column with N - 1
    degrees of freedom.

#### STDDEV_POP
-   `STDDEV_POP`

    Compute the standard deviation for a column with N degrees
    of freedom.

#### SUM
-   `SUM`

    Compute the sum for a column.

#### COUNT_IF
-   `COUNT_IF`

    Compute the total number of occurrences of `true` in a column
    of booleans. For example:

    ```sql
    SELECT COUNT_IF(A) FROM table1
    ```

    Is equivalent to
    ```sql
    SELECT SUM(CASE WHEN A THEN 1 ELSE 0 END) FROM table1
    ```

#### VARIANCE
-   `VARIANCE`

    Compute the variance for a column with N - 1 degrees of
    freedom.

#### VAR_SAMP
-   `VAR_SAMP`

    Compute the variance for a column with N - 1 degrees of
    freedom.

#### VAR_POP
-   `VAR_POP`

    Compute the variance for a column with N degrees of freedom.

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

Bodo SQL currently supports the following Timestamp functions:

#### DATEDIFF
-   `DATEDIFF(timestamp_val1, timestamp_val2)`

    Computes the difference in days between two Timestamp
    values

#### STR_TO_DATE
-   `STR_TO_DATE(str_val, literal_format_string)`

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
-   `DATE_FORMAT(timestamp_val, literal_format_string)`

    Converts a timestamp value to a String value given a
    scalar format string.

    Recognized formatting characters:

    -   `%i` Minutes, zero padded (00 to 59)
    -   `%M` Full month name (January to December)
    -   `%r` Time in format in the format (hh\:mm\:ss AM/PM)
    -   `%s` Seconds, zero padded (00 to 59)
    -   `%T` Time in format in the format (hh\:mm\:ss)
    -   `%T` Time in format in the format (hh\:mm\:ss)
    -   `%u` week of year, where monday is the first day of the week(00 to 53)
    -   `%a` Abbreviated weekday name (sun-sat)
    -   `%b` Abbreviated month name (jan-dec)
    -   `%f` Microseconds, left padded with 0's, (000000 to 999999)
    -   `%H` Hour, zero padded (00 to 23)
    -   `%j` Day Of Year, left padded with 0's (001 to 366)
    -   `%m` Month number (00 to 12)
    -   `%p` AM or PM, depending on the time of day
    -   `%d` Day of month, zero padded (01 to 31)
    -   `%Y` Year as a 4 digit value
    -   `%y` Year as a 2 digit value, zero padded (00 to 99)
    -   `%U` Week of year where sunday is the first day of the week
        (00 to 53)
    -   `%S` Seconds, zero padded (00 to 59)

    For example:

    ```sql
    DATE_FORMAT(Timestamp '2020-01-12', '%Y %m %d') =='2020 01 12'
    DATE_FORMAT(Timestamp '2020-01-12 13:39:12', 'The time was %T %p. It was a %u') =='The time was 13:39:12 PM. It was a Sunday'
    ```

#### DATE_ADD
-   `DATE_ADD(timestamp_val, interval)`

    Computes a timestamp column by adding an interval column/scalar to a
    timestamp value. If the first argument is a string representation of a
    timestamp, Bodo will cast the value to a timestamp.

#### DATE_SUB
-   `DATE_SUB(timestamp_val, interval)`

    Computes a timestamp column by subtracting an interval column/scalar
    to a timestamp value. If the first argument is a string representation
    of a timestamp, Bodo will cast the value to a timestamp.

#### DATE_TRUNC
-   `DATE_TRUNC(str_literal, timestamp_val)`

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

#### NOW
-   `NOW()`

    Computes a timestamp equal to the current system time

#### LOCALTIMESTAMP
-   `LOCALTIMESTAMP()`

    Equivalent to `NOW`

#### CURDATE
-   `CURDATE()`

    Computes a timestamp equal to the current system time, excluding the
    time information

#### CURRENT_DATE
-   `CURRENT_DATE()`

    Equivalent to `CURDATE`

#### EXTRACT
-   `EXTRACT(TimeUnit from timestamp_val)`

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

    TimeUnits are not case sensitive.

#### MICROSECOND
-   `MICROSECOND(timestamp_val)`

    Equivalent to `EXTRACT(MICROSECOND from timestamp_val)`

#### SECOND
-   `SECOND(timestamp_val)`

    Equivalent to `EXTRACT(SECOND from timestamp_val)`

#### MINUTE
-   `MINUTE(timestamp_val)`

    Equivalent to `EXTRACT(MINUTE from timestamp_val)`

#### HOUR
-   `HOUR(timestamp_val)`

    Equivalent to `EXTRACT(HOUR from timestamp_val)`

#### WEEK
-   `WEEK(timestamp_val)`

    Equivalent to `EXTRACT(WEEK from timestamp_val)`

#### WEEKOFYEAR
-   `WEEKOFYEAR(timestamp_val)`

    Equivalent to `EXTRACT(WEEK from timestamp_val)`

#### MONTH
-   `MONTH(timestamp_val)`

    Equivalent to `EXTRACT(MONTH from timestamp_val)`

#### QUARTER
-   `QUARTER(timestamp_val)`

    Equivalent to `EXTRACT(QUARTER from timestamp_val)`

#### YEAR
-   `YEAR(timestamp_val)`

    Equivalent to `EXTRACT(YEAR from timestamp_val)`

#### WEEKISO
-   `WEEKISO(timestamp_val)`

    Computes the ISO week for the provided timestamp value.

#### YEAROFWEEKISO
-   `YEAROFWEEKISO(timestamp_val)`

    Computes the ISO year for the provided timestamp value.

#### MAKEDATE
-   `MAKEDATE(integer_years_val, integer_days_val)`

    Computes a timestamp value that is the specified number of days after
    the specified year.

#### DAYNAME
-   `DAYNAME(timestamp_val)`

    Computes the string name of the day of the timestamp value.

#### MONTHNAME
-   `MONTHNAME(timestamp_val)`

    Computes the string name of the month of the timestamp value.

#### TO_DAYS
-   `TO_DAYS(timestamp_val)`

    Computes the difference in days between the input timestamp, and year
    0 of the Gregorian calendar

#### TO_SECONDS
-   `TO_SECONDS(timestamp_val)`

    Computes the number of seconds since year 0 of the Gregorian calendar

#### FROM_DAYS
-   `FROM_DAYS(n)`

    Returns a timestamp values that is n days after year 0 of the
    Gregorian calendar

#### UNIX_TIMESTAMP
-   `UNIX_TIMESTAMP()`

    Computes the number of seconds since the unix epoch

#### FROM_UNIXTIME
-   `FROM_UNIXTIME(n)`

    Returns a Timestamp value that is n seconds after the unix epoch

#### ADDDATE
-   `ADDDATE(timestamp_val, interval)`

    Same as `DATE_ADD`

#### SUBDATE
-   `SUBDATE(timestamp_val, interval)`

    Same as `DATE_SUB`

#### TIMESTAMPDIFF
-   `TIMESTAMPDIFF(unit, timestamp_val1, timestamp_val2)`

    Returns timestamp_val1 - timestamp_val2 rounded down to the provided
    unit.

#### WEEKDAY
-   `WEEKDAY(timestamp_val)`

    Returns the weekday number for timestamp_val.

    !!! note
        Monday = 0, Sunday=6

#### YEARWEEK
-   `YEARWEEK(timestamp_val)`

    Returns the year and week number for the provided timestamp_val
    concatenated as a single number. For example:
    ```sql
    YEARWEEK(TIMESTAMP '2021-08-30::00:00:00')
    202135
    ```

#### LAST_DAY
-   `LAST_DAY(timestamp_val)`

    Given a timestamp value, returns a timestamp value that is the last
    day in the same month as timestamp_val.

#### UTC_TIMESTAMP
-   `UTC_TIMESTAMP()`

    Returns the current UTC date and time as a timestamp value.

#### UTC_DATE
-   `UTC_DATE()`

    Returns the current UTC date as a Timestamp value.

#### TO_DATE
-   `TO_DATE(col_expr)`

    Casts the col_expr to a timestamp column truncated to the date
    portion. Supported for Integers, Strings, and Datetime types.
    For information on valid for conversion, see: https://docs.snowflake.com/en/sql-reference/functions/to_date.html.
    Raises an error if suplied an invalid expression.

#### TRY_TO_DATE
-   `TRY_TO_DATE(col_expr)`

    See TO_DATE. The only difference is that TRY_TO_DATE will return NULL upon encountering an invalid expression
    NULL instead of raising an error. We recommend using this function for converting to date.

###  String Functions

Bodo SQL currently supports the following string functions:

#### LOWER
-   `LOWER(str)`

    Converts the string scalar/column to lower case.

#### LCASE
-   `LCASE(str)`

    Same as `LOWER`.

#### UPPER
-   `UPPER(str)`

    Converts the string scalar/column to upper case.

#### UCASE
-   `UCASE(str)`

    Same as `UPPER`.

#### CONCAT
-   `CONCAT(str_0, str_1, ...)`

    Concatenates the strings together. Requires at least two
    arguments.

#### CONCAT_WS
-   `CONCAT_WS(str_separator, str_0, str_1, ...)`

    Concatenates the strings together, with the specified
    separator. Requires at least three arguments

#### SUBSTRING
-   `SUBSTRING(str, start_index, len)`

    Takes a substring of the specified string, starting at the
    specified index, of the specified length. `start_index = 1`
    specifies the first character of the string, `start_index =
    -1` specifies the last character of the string. `start_index
    = 0` causes the function to return empty string. If
    `start_index` is positive and greater then the length of the
    string, returns an empty string. If `start_index` is
    negative, and has an absolute value greater then the
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
-   `MID(str, start_index, len)`

    Equivalent to `SUBSTRING`

#### SUBSTR
-   `SUBSTR(str, start_index, len)`

    Equivalent to `SUBSTRING`

#### LEFT
-   `LEFT(str, n)`

    Takes a substring of the specified string consisting of
    the leftmost n characters

#### RIGHT
-   `RIGHT(str, n)`

    Takes a substring of the specified string consisting of
    the rightmost n characters

#### REPEAT
-   `REPEAT(str, len)`

    Extends the specified string to the specified length by
    repeating the string. Will truncate the string If the
    string's length is less then the len argument

    For example:

    ```sql
    REPEAT('abc', 7) =='abcabca'
    REPEAT('hello world', 5) =='hello'
    ```


#### STRCMP
-   `STRCMP(str1, str2)`

    Compares the two strings lexicographically. If `str1 > str2`,
    return 1. If `str1 < str2`, returns -1. If `str1 == str2`,
    returns 0.

#### REVERSE
-   `REVERSE(str)`

    Returns the reversed string.

#### ORD
-   `ORD(str)`

    Returns the integer value of the unicode representation of
    the first character of the input string. returns 0 when
    passed the empty string

#### CHAR
-   `CHAR(int)`

    Returns the character of the corresponding unicode value.
    Currently only supported for ASCII characters (0 to 127,
    inclusive)

#### SPACE
-   `SPACE(int)`

    Returns a string containing the specified number of
    spaces.

#### LTRIM
-   `LTRIM(str)`

    returns the input string, will remove all spaces from the
    left of the string

#### RTRIM
-   `RTRIM(str)`

    returns the input string, will remove all spaces from the
    right of the string

#### TRIM
-   `TRIM(str)`

    returns the input string, will remove all spaces from the
    left and right of the string

#### SUBSTRING_INDEX
-   `SUBSTRING_INDEX(str, delimiter_str, n)`

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
-   `LPAD(string, len, padstring)`

    Extends the input string to the specified length, by
    appending copies of the padstring to the left of the
    string. If the input string's length is less then the len
    argument, it will truncate the input string.

    For example:
    ```sql
    LPAD('hello', 10, 'abc') =='abcabhello'
    LPAD('hello', 1, 'abc') =='h'
    ```

#### RPAD
-   `RPAD(string, len, padstring)`

    Extends the input string to the specified length, by
    appending copies of the padstring to the right of the
    string. If the input string's length is less then the len
    argument, it will truncate the input string.

    For example:
    ```sql
    RPAD('hello', 10, 'abc') =='helloabcab'
    RPAD('hello', 1, 'abc') =='h'
    ```


#### REPLACE
-   `REPLACE(base_string, substring_to_remove, string_to_substitute)`

    Replaces all occurrences of the specified substring with
    the substitute string.

    For example:
    ```sql
    REPLACE('hello world', 'hello' 'hi') =='hi world'
    ```


#### LENGTH
-   `LENGTH(string)`

    Returns the number of characters in the given string.


#### EDITDISTANCE
-   `EDITDISTANCE(string0, string1[, max_distance])`

    Returns the minimum edit distance between string0 and string1
    according to Levenshtein distance. Optionally accepts a third
    argument specifying a maximum distance value. If the minimum
    edit distance between the two strings exceeds this value, then
    this value is returned instead. If it is negative, zero
    is returned.


#### SPLIT_PART
-   `SPLIT_PART(source, delimeter, part)`

    Returns the substring of the source between certain occurrence of
    the delimeter string, the occurrence being specified by the part.
    I.e. if part=1, returns the substring before the first occurrence,
    and if part=2, returns the substring between the first and second
    occurrence. Zero is treated like 1. Negative indicies are allowed.
    If the delimeter is empty, the source is treated like a single token.
    If the part is out of bounds, '' is returned.


 #### STRTOK
-   `STRTOK(source[, delimeter[, part]])`

    Tokenizes the source string by occurrences of any character in the
    delimeter string and returns the occurrence specified by the part.
    I.e. if part=1, returns the substring before the first occurrence,
    and if part=2, returns the substring between the first and second
    occurrence. Zero and negative indices are not allowed. Empty tokens
    are always skipped in favor of the next non-empty token. In any
    case where the only possible output is '', the output is `NULL`.
    The delimeter is optional and defaults to ' '. The part is optional
    and defaults to 1.


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

* Currently, extra backslashes may be required to escape certain characters if they have meaning in Python. The amount of backslashes required to properly escape a character depends on the useage.

* All matches are non-overlapping.

* If any of the numeric arguments are zero or negative, or the `group_num` argument is out of bounds, an error is raised. The only exception is `REGEXP_REPLACE`, which allows its occurrence argument to be zero.

BodoSQL currently supports the following regex functions:

#### REGEXP_LIKE
-   `REGEXP_LIKE(str, pattern[, flag])`

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

    - 3 arguments: Returns `true` if `A` starts with the letters `'THE'` (case-insenstive).
    ```sql
    SELECT REGEXP_LIKE(A, 'THE.*', 'i')
    ```


#### REGEXP_COUNT
-   `REGEXP_COUNT(str, pattern[, position[, flag]])`

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

    - 4 arguments: Returns the number of times that a substring occurrs in `A` that contains two
    ones with any character (including newlines) in between.
    ```sql
    SELECT REGEXP_COUNT(A, '1.1', 1, 's')
    ```


#### REGEXP_REPLACE
-   `REGEXP_REPLACE(str, pattern[, replacement[, position[, occurrence[, flag]]]])`

    Returns the a version of the inputted string where each
    match to the pattern is replaced by the replacement string,
    starting at the location specified by the `position` argument
    (with 1-indexing). The occurrence argument specifies which
    match to replace, where 0 means replace all occurrences. If
    `replacement` is not provided, `''` is used. If `position` is
    not provided, `1` is used. If `occurrence` is not provided,
    `0` is used. If `flag` is not provided, `''` is used.

    If there are an insufficient number of matches, or the pattern is empty,
    the original string is returned.

    Note: backreferences in the replacement pattern are supported,
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
-   `REGEXP_SUBSTR(str, pattern[, position[, occurrence[, flag[, group_num]]]])`

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
-   `REGEXP_INSTR(str, pattern[, position[, occurrence[, option[, flag[, group_num]]]]])`

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


###   Control Flow Functions

#### DECODE
-   `DECODE(Arg0, Arg1, Arg2, ...)`

    When `Arg0` is `Arg1`, outputs `Arg2`. When `Arg0` is `Arg3`,
    outputs `Arg4`. Repeats until it runs out of pairs of arguments.
    At this point, if there is one remaining argument, this is used
    as a default value. If not, then the output is NULL. Note: treats
    NULL as a literal value that can be matched on.

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
-   `EQUAL_NULL(A, B)`

    Returns true if A and B are both `NULL`, or both non-null and
    equal to each other.


#### IF
-   `IF(Cond, TrueValue, FalseValue)`

    Returns `TrueValue` if cond is true, and `FalseValue` if cond is
    false. Logically equivalent to:

    ```sql
    CASE WHEN Cond THEN TrueValue ELSE FalseValue END
    ```


#### IFF
-   `IFF(Cond, TrueValue, FalseValue)`

    Equivalent to `IF`


#### IFNULL
-   `IFNULL(Arg0, Arg1)`

    Returns `Arg1` if `Arg0` is `null`, and otherwise returns `Arg1`. If
    arguments do not have the same type, Bodo SQL will attempt
    to cast them all to a common type, which is currently
    undefined behavior.

#### ZEROIFNULL
-   `ZEROIFNULL(Arg0, Arg1)`

    Equivalent to `IFNULL(Arg0, 0)`


#### NVL
-   `NVL(Arg0, Arg1)`

    Equivalent to `IFNULL`


#### NVL2
-   `NVL2(Arg0, Arg1, Arg2)`

    Equivalent to `NVL(NVL(Arg0, Arg1), Arg2)`


#### NULLIF
-   `NULLIF(Arg0, Arg1)`

    Returns `null` if the `Arg0` evaluates to true, and otherwise
    returns `Arg1`


#### NULLIFZERO
-   `NULLIFZERO(Arg0)`

    Equivalent to `NULLIF(Arg0, 0)`


#### COALESCE
-   `COALESCE(A, B, C, ...)`

    Returns the first non NULL argument, or NULL if no non NULL
    argument is found. Requires at least two arguments. If
    Arguments do not have the same type, Bodo SQL will attempt
    to cast them to a common dtype, which is currently
    undefined behavior.


### Window Functions

Window functions can be used to compute an aggregation across a
row and its surrounding rows. Most window functions have the
following syntax:

```sql
SELECT WINDOW_FN(ARG1, ..., ARGN) OVER (PARTITION BY PARTITION_COLUMN_1, ..., PARTITION_COLUMN_N ORDER BY SORT_COLUMN_1, ..., SORT_COLUMN_N ROWS BETWEEN <LOWER_BOUND> AND <UPPER_BOUND>) FROM table_name
```
The `ROWS BETWEEN ROWS BETWEEN <LOWER_BOUND> AND <UPPER_BOUND>`
section is used to specify the window over which to compute the
function. A bound can can come before the current row, using
`PRECEDING` or after the current row, using
`FOLLOWING`. The bounds can be relative (i.e.
`N PRECEDING` or `N FOLLOWING`), where `N` is a positive integer,
or they can be absolute (i.e. `UNBOUNDED PRECEDING` or
`UNBOUNDED FOLLOWING`).

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

Window functions perform a series of steps as followed:

1.  Partition the data by `PARTITION_COLUMN`. This is effectively a groupby operation on `PARTITION_COLUMN`.
2.  Sort each group as specified by the `ORDER BY` clause.
3.  Perform the calculation over the specified window, i.e. the newly ordered subset of data.
4.  Shuffle the data back to the original ordering.

For BodoSQL, `PARTITION BY` is required, but
`ORDER BY` is optional for most functions and
`ROWS BETWEEN` is optional for all of them. If
`ROWS BETWEEN` is not specified then it defaults to either
computing the result over the entire window (if no `ORDER BY`
clause is specified) or to using the window `UNBOUNDED PRECEDING TO CURRENT ROW`
(if there is an `ORDER BY` clause).
Note: `RANGE BETWEEN` is not currently supported.
Currently BodoSQL supports the following Window functions:


#### COUNT
-   `COUNT(*)`

    Compute the number of entries in a window.

#### SUM
-   `SUM(COLUMN_EXPRESSION)`

    Compute the sum over the window or `NULL` if the window is
    empty.

#### AVG
-   `AVG(COLUMN_EXPRESSION)`

    Compute the average over the window or `NULL` if the window
    is empty.


#### STDDEV
-   `STDDEV(COLUMN_EXPRESSION)`

    Compute the standard deviation for a sample over the
    window or `NULL` if the window is empty.

#### STDDEV_POP
-   `STDDEV_POP(COLUMN_EXPRESSION)`

    Compute the standard deviation for a population over the
    window or `NULL` if the window is empty.

#### VARIANCE
-   `VARIANCE(COLUMN_EXPRESSION)`

    Compute the variance for a sample over the window or `NULL`
    if the window is empty.

#### VAR_POP
-   `VAR_POP(COLUMN_EXPRESSION)`

    Compute the variance for a population over the window or
    `NULL` if the window is empty.

#### MAX
-   `MAX(COLUMN_EXPRESSION)`

    Compute the maximum value over the window or NULL if the
    window is empty.

#### MIN
-   `MIN(COLUMN_EXPRESSION)`

    Compute the minimum value over the window or NULL if the
    window is empty.

#### COUNT

-   `COUNT(COLUMN_EXPRESSION)`

    Compute the number of non-NULL entries in a window.

#### COUNT_IF

-   `COUNT_IF(BOOLEAN_COLUMN_EXPRESSION)`

    Compute the number of `true` entries in a boolean column.


#### LEAD
-   `LEAD(COLUMN_EXPRESSION, [N], [FILL_VALUE])`

    Returns the row that follows the current row by N. If N
    is not specified, defaults to 1. If FILL_VALUE is not
    specified, defaults to `NULL`. If
    there are fewer than N rows the follow the current row in
    the window, it returns FILL_VALUE. N must be a literal
    non-negative integer if specified. FILL_VALUE must be a
    scalar if specified. Note: at this time Bodo does not
    support the `IGNORE NULLS` keyword.

    This function cannot be used with `ROWS BETWEEN`.

#### LAG
-   `LAG(COLUMN_EXPRESSION, [N], [FILL_VALUE])`

    Returns the row that precedes the current row by N. If N
    is not specified, defaults to 1. If FILL_VALUE is not
    specified, defaults to `NULL`. If
    there are fewer than N rows the precede the current row in
    the window, it returns FILL_VALUE. N must be a literal
    non-negative integer if specified. FILL_VALUE must be a
    scalar if specified. Note: at this time Bodo does not
    support the `IGNORE NULLS` keyword.

    This function cannot be used with `ROWS BETWEEN`.

#### FIRST_VALUE
-   `FIRST_VALUE(COLUMN_EXPRESSION)`

    Select the first value in the window or `NULL` if the window
    is empty.

#### LAST_VALUE
-   `LAST_VALUE(COLUMN_EXPRESSION)`

    Select the last value in the window or `NULL` if the window
    is empty.

#### NTH_VALUE
-   `NTH_VALUE(COLUMN_EXPRESSION, N)`

    Select the Nth value in the window (1-indexed) or `NULL` if
    the window is empty. If N is greater or than the window
    size, this returns `NULL`.


#### ANY_VALUE
-   `ANY_VALUE(COLUMN_EXPRESSION)`

    Select an arbitrary value in the window or `NULL` if the window
    is empty. Note: currently BodoSQL always selects the first value,
    but this is subject to change at any time.


#### NTILE
-   `NTILE(N)`

    Divides the partitioned groups into N buckets based on
    ordering. For example if N=3 and there are 30 rows in a
    partition, the first 10 are assigned 1, the next 10 are
    assigned 2, and the final 10 are assigned 3.

#### RANK
-   `RANK()`


    Compute the rank of each row based on the value(s) in the row relative to all value(s) within the partition.
    The rank begins with 1 and increments by one for each succeeding value. Duplicate value(s) will produce
    the same rank, producing gaps in the rank (compare with `DENSE_RANK`). `ORDER BY` is required for this function.


#### DENSE_RANK
-   `DENSE_RANK()`

    Compute the rank of each row based on the value(s) in the row relative to all value(s) within the partition
    without producing gaps in the rank (compare with `RANK`). The rank begins with 1 and increments by one for each succeeding value.
    Rows with the same value(s) produce the same rank. `ORDER BY` is required for this function.

!!!note
    To compare `RANK` and `DENSE_RANK`, on input array `['a', 'b', 'b', 'c']`, `RANK` will output `[1, 2, 2, 4]` while `DENSE_RANK` outputs `[1, 2, 2, 3]`.

#### PERCENT_RANK
-   `PERCENT_RANK()`

    Compute the percentage ranking of the value(s) in each row based on the value(s) relative to all value(s)
    within the window partition. Ranking calcuated using `RANK()` divided by the number of rows in the window
    partition minus one. Paritions with one row have `PERCENT_RANK()` of 0. `ORDER BY` is required for this function.


#### CUME_DIST
-   `CUME_DIST()`

    Compute the cumulative distribution of the value(s) in each row based on the value(s) relative to all value(s)
    within the window partition. `ORDER BY` is required for this function.


#### ROW_NUMBER
-   `ROW_NUMBER()`

    Compute an increasing row number (starting at 1) for each
    row. This function cannot be used with `ROWS BETWEEN`.


#### CONDITIONAL_TRUE_EVENT
-   `CONDITIONAL_TRUE_EVENT(BOOLEAN_COLUMN_EXPRESSION)`

    Computes a counter within each partition that starts at zero and increases by 1 each
    time the boolean column's value is `true`. `ORDER BY` is required for this function.


#### CONDITIONAL_CHANGE_EVENT
-   `CONDITIONAL_CHANGE_EVENT(COLUMN_EXPRESSION)`

    Computes a counter within each partition that starts at zero and increases by 1 each
    time the value inside the window changes. `NULL` does not count as a new/changed value.
    `ORDER BY` is required for this function.

 
### Casting / Conversion Functions

BodoSQL currently supports the following casting/conversion functions:

#### TO_BOOLEAN
-  `TO_BOOLEAN(COLUMN_EXPRESSION)`

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
-  `TRY_TO_BOOLEAN(COLUMN_EXPRESSION)`

    This is similar to `TO_BOOLEAN` except that it will return `NULL` instead of throwing an error invalid inputs.
    
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

-   `boolean_literal`
-   `datetime_literal`
-   `float_literal`
-   `integer_literal`
-   `interval_literal`
-   `string_literal`

### Boolean Literal {#boolean_literal}

**Syntax**:

```sql
TRUE | FALSE
```

Boolean literals are case insensitive.

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

```
DAY[S] | HOUR[S] | MINUTE[S] | SECOND[S]
```

In addition we also have limited support for `YEAR[S]` and `MONTH[S]`.
These literals cannot be stored in columns and currently are only
supported for operations involving add and sub.

### String Literal {#string_literal}

**Syntax**:

```sql
'char [ ... ]'
```

Where char is a character literal in a Python string.

## NULL Semantics

Bodo SQL converts SQL queries to Pandas code that executes inside Bodo.
As a result, NULL behavior aligns with Pandas and may be slightly
different than other SQL systems. This is currently an area of active
development to ensure compatibility with other SQL systems.

Most operators with a `NULL` input return `NULL`. However, there a couple
notable places where Bodo SQL may not match other SQL systems:

-   Bodo SQL treats `NaN` the same as `NULL`
-   Is (NOT) False and Is (NOT) True return `NULL` when used on a null
    expression
-   AND will return `NULL` if any of the inputs is `NULL`

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

    Note: This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
    e.g. `bc = bc.add_or_replace_view("t1", table)`



- ++bodosql.BodoSQLContext.%%remove_view%%(self, name: str)++
<br><br>

    Creates a new `BodoSQLContext` from an existing context by removing the table with the
    given name. If the name does not exist, a `BodoError` is thrown.

    ***Arguments***

    - `name`: The name of the table to remove.

    ***Returns***

    A new `BodoSQLContext` that retains the tables and catalogs from the old `BodoSQLContext` minus the table specified.

    Note: This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
    e.g. `bc = bc.remove_view("t1")`


- ++bodosql.BodoSQLContext.%%add_or_replace_catalog%%(self, catalog: DatabaseCatalog)++
<br><br>

    Create a new `BodoSQLContext` from an existing context by replacing the `BodoSQLContext` object's `DatabaseCatalog` with
    a new catalog.

    ***Arguments***

    - `catalog`: The catalog to insert.

    ***Returns***

    A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but replaces the old catalog with the new catalog specified.

    Note: This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
    e.g. `bc = bc.add_or_replace_catalog(catalog)`


- ++bodosql.BodoSQLContext.%%remove_catalog%%(self)++
<br><br>

    Create a new `BodoSQLContext` from an existing context by removing its `DatabaseCatalog`.

    ***Returns***

    A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but removes the old catalog.

    Note: This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
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

Currently BodoSQL supports catalogs Snowflake, but support other data storage systems will be added in future releases.

### SnowflakeCatalog

The Snowflake Catalog offers an interface for users to connect their Snowflake accounts to use with BodoSQL.
With a Snowflake Catalog, users only have to specify their Snowflake connection once, and can access any tables of interest in their Snowflake account. Currently the Snowflake Catalog is defined to
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

  * SELECT
  * INSERT INTO
  * DELETE
