.. _bodosql:

Bodo SQL
========

Bodo SQL provides seamless integration of pandas programs with SQL queries on structured data. You can use it to either
port your existing SQL code to pandas programs, or write pandas programs including SQL queries passed to the Bodo SQL context.
The result of executing SQL code with Bodo SQL is returned as a DataFrame.


Bodo SQL context
----------------

For those familiar with Spark SQL, Bodo SQL offers a similar interface with SQL queries between pandas through Bodo SQL context.
DataFrames are created and accessed through Bodo SQL context::

    import bodosql
    import pytest

    df1 = pd.DataFrame(
        {"A": [1, 3, 2, 1], "B": [2.2, 1.2, 4.4, 2.3], "C": [5, 2, 1, 4]}
    )
    df2 = pd.DataFrame({"A": [4, 1, 2], "D": [5.1, 5.2, 2.3]})
    test_df2['DateCol']  = pd.to_datetime(test_df2['DateCol'])
    bc = bodosql.BodoSQLContext({"table1": df1, "table2": df2})

SQL queries can then be passed to the Bodo SQL context::

    query = "select A from table1"
    df1 = bc.sql(query)
    print(df1)

Bodo SQL automatically translates this query into its corresponding pandas code::

  df1 = table1[['A',]]
  df2 = df1.rename(copy=False,columns={'A':'a',})

which returns a dataframe::

       a
    0  1
    1  3
    2  2
    3  1


How it works
------------

Bodo SQL uses Apache Calcite to parse SQL queries and produce a Physical Plan, which is then used to generate pandas code which
performs step-by-step operations to compute the final DataFrame.

Supported Operations
--------------------
We currently support the following SQL query statements and clauses with Bodo SQL, and are continuously adding support towards completeness. Note that
currently, Bodo SQL ignores casing of keywords, and column and table names. So, `select a from table1` is treated the same as `SELECT A FROM TABLE1`.

#. :func:`SELECT`

The `SELECT` statement is used to select data in the form of columns. The data returned from Bodo SQL is stored in a DataFrame. Example usage::

    SELECT <COLUMN_NAMES> FROM <TABLE_NAME>

The `SELECT DISTINCT` statement is used to return only distinct (different) values::

    SELECT DISTINCT <COLUMN_NAMES> FROM <TABLE_NAME>

#. :func:`WHERE`
The `WHERE` clause on columns can be used to filter records that satisfy specific conditions::

    SELECT <COLUMN_NAMES> FROM <TABLE_NAME> WHERE <CONDITION>

Note that while the `WHERE` clause can also be used in `UPDATE` and `DELETE` statements, which we will support in a future release.

#. :func:`ORDER BY`
The `ORDER BY` keyword sorts the resulting DataFrame in ascending or descending order. By default, it sorts the records in ascending order.
For descending order, we use the `DESC` keyword::

    SELECT <COLUMN_NAMES>
    FROM <TABLE_NAME>
    ORDER BY <ORDERED_COLUMN_NAMES> ASC|DESC

#. Null Values

We can check for null values using the `IS_NULL` and `IS_NOT_NULL` keywords::

    SELECT <COLUMN_NAMES>
    FROM <TABLE_NAME>
    WHERE <COLUMN_NAME> IS NULL


    SELECT <COLUMN_NAMES>
    FROM <TABLE_NAME>
    WHERE <COLUMN_NAME> IS NOT NULL

#. :func:`LIMIT`

Bodo SQL supports the `LIMIT` keyword to select a limited number of rows::

    SELECT <COLUMN_NAMES>
    FROM <TABLE_NAME>
    WHERE <CONDITION>
    LIMIT <NUMBER>

#. Aggregation Functions

  - The `MIN()`, and `MAX()` functions return the smallest and the largest value of the selected column respectively::

        SELECT MIN(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

        SELECT MAX(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

  - The `COUNT()` function can be used to count the number of rows that match a condition::

        SELECT COUNT(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

    The `SUM()` function returns the total sum of a column with numeric values.

        SELECT SUM(<COLUMN_NAME>)
        FROM <TABLE_NAME>
        WHERE <CONDITION>;

#. :func:`IN`
The IN keyword is used to pick specific values of a column in a WHERE clause::

    SELECT <COLUMN_NAMES>
    FROM <TABLE_NAME>
    WHERE <COLUMN_NAME> IN <VALUES>;


    SELECT <COLUMN_NAMES>
    FROM <TABLE_NAME>
    WHERE <COLUMN_NAME> IN (SELECT STATEMENT);
