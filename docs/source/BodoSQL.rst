.. _bodosql:

Bodo SQL
========

Bodo SQL provides high performance and scalable SQL query execution
using Bodo's HPC capabilities and optimizations.
It also provides native Python/SQL integration
as well as SQL to Pandas conversion for the first time.
BodoSQL is in early stages and its capabilities are expanding rapidly.


Using Bodo SQL
--------------

The example below demonstrates using Bodo SQL in Python programs.
It loads data into a dataframe, runs a SQL query on the data,
and runs Python/Pandas code on query results::


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


This program is fully type checked, optimized and parallelized by Bodo end-to-end.
`BodoSQLContext` creates a SQL environment with tables created from dataframes.
`BodoSQLContext.sql()` runs a SQL query and returns the results as a dataframe.
`BodoSQLContext` can be used outside Bodo JIT functions if necessary as well.


SQL to Pandas Conversion
------------------------

Bodo SQL can generate Pandas code from SQL queries automatically. For example::

    bc.convert_to_pandas("SELECT A FROM table1 WHERE B > 4")

returns::

    df1 = table1[["A","B",]][(table1["B"] > 4)]
    df2 = pd.DataFrame({"A": df1["A"], })
    return df2


Using Python/Pandas code instead of SQL can simplify existing data science applications
and improve code maintenance.


Supported Operations
--------------------
We currently support the following SQL query statements and clauses with Bodo SQL, and are continuously adding support towards completeness. Note that
Bodo SQL ignores casing of keywords, and column and table names. Therefore, ``select a from table1`` is treated the same as ``SELECT A FROM TABLE1``.

* `SELECT`

    The ``SELECT`` statement is used to select data in the form of columns. The data returned from Bodo SQL is stored in a dataframe. Example usage::

        SELECT <COLUMN_NAMES> FROM <TABLE_NAME>

    The ``SELECT DISTINCT`` statement is used to return only distinct (different) values::

        SELECT DISTINCT <COLUMN_NAMES> FROM <TABLE_NAME>

* `WHERE`

    The ``WHERE`` clause on columns can be used to filter records that satisfy specific conditions::

        SELECT <COLUMN_NAMES> FROM <TABLE_NAME> WHERE <CONDITION>


* `ORDER BY`

    The ``ORDER BY`` keyword sorts the resulting dataframe in ascending or descending order. By default, it sorts the records in ascending order.
    For descending order, the ``DESC`` keyword can be used::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        ORDER BY <ORDERED_COLUMN_NAMES> ASC|DESC


* Null Values

    ``IS NULL`` and ``IS NOT NULL`` conditions check for null values::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> IS NULL


        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> IS NOT NULL


* `LIMIT`

    Bodo SQL supports the ``LIMIT`` keyword to select a limited number of rows::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <CONDITION>
        LIMIT <NUMBER>


* Aggregation Functions

  - The ``MIN()``, and ``MAX()`` functions return the smallest and the largest value of the selected column respectively::

        SELECT MIN(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

        SELECT MAX(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

  - The ``COUNT()`` function can be used to count the number of rows that match a condition::

        SELECT COUNT(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

  - The ``SUM()`` function returns the total sum of a column with numeric values::

        SELECT SUM(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

  - The AVG() function returns the average value of a numeric column::

        SELECT AVG(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;


* `IN`

    The ``IN`` keyword is used to pick specific values of a column in a ``WHERE`` clause::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> IN <VALUES>;


        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> IN (SELECT STATEMENT);


* `BETWEEN`

    The ``BETWEEN`` operator selects values within a given range. The values can be numbers, text, or dates.
    The ``BETWEEN`` operator is inclusive: begin and end values are included::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> BETWEEN <VALUE1> AND <VALUE2>;


* `JOIN`

    A ``JOIN`` clause is used to combine rows from two or more tables, based on a related column between them::

      SELECT <COLUMN_NAMES>
        FROM <LEFT_TABLE_NAME>
        <JOIN_TYPE> <RIGHT_TABLE_NAME>
        ON <LEFT_TABLE_COLUMN_NAME> = <RIGHT_TABLE_COLUMN_NAME>;

    Here are the different types of the joins in SQL:

    - ``(INNER) JOIN``: returns records that have matching values in both tables
    - ``LEFT (OUTER) JOIN``: returns all records from the left table, and the matched records from the right table
    - ``RIGHT (OUTER) JOIN``: returns all records from the right table, and the matched records from the left table
    - ``FULL (OUTER) JOIN``: returns all records when there is a match in either left or right table


* `UNION`
    The ``UNION`` operator is used to combine the result-set of two or more ``SELECT`` statements::

        SELECT <COLUMN_NAMES> FROM <TABLE1>
        UNION
        SELECT <COLUMN_NAMES> FROM <TABLE2>;

    Each ``SELECT`` statement within ``UNION`` must have the same number of columns.
    The columns must also have similar data types, and columns in each ``SELECT`` statement must also be in the same order.


    The ``UNION`` operator selects only distinct values by default. To allow duplicate values, use ``UNION ALL``::

        SELECT <COLUMN_NAMES> FROM <TABLE1>
        UNION ALL
        SELECT <COLUMN_NAMES> FROM <TABLE2>;


* `GROUP BY`
    The ``GROUP BY`` statement groups rows that have the same values into summary rows, like "find the number of customers in each country".
    The ``GROUP BY`` statement is often used with aggregate functions (``COUNT``, ``MAX``, ``MIN``, ``SUM``, ``AVG``) to group the result-set by one or more columns::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <CONDITION>
        GROUP BY <COLUMN_NAMES>
        ORDER BY <COLUMN_NAMES>;


* `HAVING`
    The `HAVING` clause was added to SQL because the WHERE keyword could not be used with aggregate functions::

        SELECT column_name(s)
        FROM table_name
        WHERE condition
        GROUP BY column_name(s)
        HAVING condition
        ORDER BY column_name(s);


* Operators

    - Bodo SQL currently supports the following arithmetic operators on columns:

        - ``+`` (addition)
        - ``-`` (subtraction)
        - ``*`` (multiplication)

    - Bodo SQL currently supports the following comparision operators on columns:

        - ``=``	(equal to)
        - ``>``	(greater than)
        - ``<``	(less than)
        - ``>=`` (greater than or equal t)o
        - ``<=`` (less than or equal to)
        - ``<>`` (not equal to)

* Aliasing

    SQL aliases are used to give a table, or a column in a table, a temporary name::

        SELECT <COLUMN_NAME> AS <ALIAS>
        FROM <TABLE_NAME>;

    Aliases are often used to make column names more readable. An alias only exists for the duration of the query.
