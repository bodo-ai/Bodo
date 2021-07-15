.. _bodosql:

Bodo SQL
========

Bodo SQL provides high performance and scalable SQL query execution
using Bodo's HPC capabilities and optimizations.
It also provides native Python/SQL integration
as well as SQL to Pandas conversion for the first time.
BodoSQL is in early stages and its capabilities are expanding rapidly.


Getting Started
---------------

Installation
~~~~~~~~~~~~
Bodo SQL is currently in an Alpha release. Please us if you are interested
in becoming a trial user.

Using Bodo SQL
~~~~~~~~~~~~~~

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
``BodoSQLContext`` creates a SQL environment with tables created from dataframes.
``BodoSQLContext.sql()`` runs a SQL query and returns the results as a dataframe.
``BodoSQLContext`` can be used outside Bodo JIT functions if necessary as well.


You can run this example by creating ``my_data.pq``::


    import pandas as pd
    import numpy as np

    NUM_GROUPS = 30
    NUM_ROWS = 20_000_000
    df = pd.DataFrame({
        "A": np.arange(NUM_ROWS) % NUM_GROUPS,
        "B": np.arange(NUM_ROWS)
    })
    df.to_parquet("my_data.pq")



SQL to Pandas Conversion
------------------------

Bodo SQL can generate Pandas code from SQL queries automatically. To view the code generated,
you can use the ``convert_to_pandas`` method, which returns the generated code as a string.
For example::

    print(bc.convert_to_pandas("SELECT A FROM table1 WHERE B > 4"))

outputs::

    def impl(table1):
        df1 = table1[(table1["B"] > np.int32(4))]
        df2 = pd.DataFrame({"A": df1["A"], })
        return df2


Using Python/Pandas code instead of SQL can simplify existing data science applications
and improve code maintenance.

**Note**: ``convert_to_pandas`` can only be executed outside Bodo JIT functions.


Aliasing
--------
In all but the most trivial cases, Bodo SQL generates internal names to avoid conflicts in the
intermediate dataframes. By default, Bodo SQL does not rename the columns for the final output
of a query using a consistent approach. For example the query::

    bc.sql("SELECT SUM(A) FROM table1 WHERE B > 4")

Results in an output column named ``$EXPR0``. To reliably reference this column
later in your code, we highly recommend using aliases for all columns that
are the final outputs of a query, such as::

    bc.sql("SELECT SUM(A) as sum_col FROM table1 WHERE B > 4")

**Note**: BodoSQL supports using aliases generated in ``SELECT`` inside ``GROUP BY``
and ``HAVING`` in the same query, but you cannot do so with ``WHERE``.

Supported Operations
--------------------
We currently support the following SQL query statements and clauses with Bodo SQL, and are continuously adding support towards completeness. Note that
Bodo SQL ignores casing of keywords, and column and table names. Therefore, ``select a from table1`` is treated the same as ``SELECT A FROM TABLE1``.

* `SELECT`

    The ``SELECT`` statement is used to select data in the form of columns. The data returned from Bodo SQL is stored in a dataframe. Example usage::

        SELECT <COLUMN_NAMES> FROM <TABLE_NAME>

    For Example::

        SELECT A FROM TABLE1

    The ``SELECT DISTINCT`` statement is used to return only distinct (different) values::

        SELECT DISTINCT <COLUMN_NAMES> FROM <TABLE_NAME>

    ``DISTINCT`` can be used in a SELECT statement or inside an aggregate function. For example::

        SELECT DISTINCT A FROM TABLE1

        SELECT COUNT DISTINCT A FROM TABLE1


* `WHERE`

    The ``WHERE`` clause on columns can be used to filter records that satisfy specific conditions::

        SELECT <COLUMN_NAMES> FROM <TABLE_NAME> WHERE <CONDITION>

    For Example::

        SELECT A FROM TABLE1 WHERE B > 4


* `ORDER BY`

    The ``ORDER BY`` keyword sorts the resulting dataframe in ascending or descending order. By default, it sorts the records in ascending order.
    For descending order, the ``DESC`` keyword can be used::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        ORDER BY <ORDERED_COLUMN_NAMES> ASC|DESC

    For Example::

        SELECT A, B FROM TABLE1 ORDER BY B, A DESC


* `LIMIT`

    Bodo SQL supports the ``LIMIT`` keyword to select a limited number of rows::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <CONDITION>
        LIMIT <NUMBER>

    For Example::

        SELECT A FROM TABLE1 LIMIT 5


* `BETWEEN`

    The ``BETWEEN`` operator selects values within a given range. The values can be numbers, text, or datetimes.
    The ``BETWEEN`` operator is inclusive: begin and end values are included::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> BETWEEN <VALUE1> AND <VALUE2>

    For example::

        Select A from Table1 where A between 10 and 100


* `JOIN`

    A ``JOIN`` clause is used to combine rows from two or more tables, based on a related column between them::

      SELECT <COLUMN_NAMES>
        FROM <LEFT_TABLE_NAME>
        <JOIN_TYPE> <RIGHT_TABLE_NAME>
        ON <LEFT_TABLE_COLUMN_NAME> = <RIGHT_TABLE_COLUMN_NAME>


    For example::

        Select table1.A, table1.B from table1 join table2 on table1.A = table2.C

    Here are the different types of the joins in SQL:

    - ``(INNER) JOIN``: returns records that have matching values in both tables
    - ``LEFT (OUTER) JOIN``: returns all records from the left table, and the matched records from the right table
    - ``RIGHT (OUTER) JOIN``: returns all records from the right table, and the matched records from the left table
    - ``FULL (OUTER) JOIN``: returns all records when there is a match in either left or right table

    Bodo SQL currently support inner join on all conditions, but all outer joins are only support on an
    equality between columns.


* `GROUP BY`
    The ``GROUP BY`` statement groups rows that have the same values into summary rows, like "find the number of customers in each country".
    The ``GROUP BY`` statement is often used with aggregate functions to group the result-set by one or more columns::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <CONDITION>
        GROUP BY <COLUMN_NAMES>
        ORDER BY <COLUMN_NAMES>

    For example::

        Select MAX(A) from table1 Group By B

    ``GROUP BY`` statements also referring to columns by alias or column number::

        Select MAX(A), B - 1 as val from table1 Group By val
        Select MAX(A), B from table1 Group By 2


* `HAVING`

    The ``HAVING`` clause is used for filtering with ``GROUP BY``. ``HAVING``
    applies the filter after generating the groups, whereas ``WHERE`` applies
    the filter before generating any groups::

        SELECT column_name(s)
        FROM table_name
        WHERE condition
        GROUP BY column_name(s)
        HAVING condition

    For example::

        Select MAX(A) from table1 Group By B HAVING C < 0

    ``HAVING`` statements also referring to columns by aliases used in the ``GROUP BY``::

        Select MAX(A), B - 1 as val from table1 Group By val having val > 5

* `CASE`

    The ``CASE`` statement goes through conditions and returns a value when the first condition is met::

        SELECT CASE WHEN cond1 THEN value1 WHEN cond2 THEN value2 ... ELSE valueN END

    For example::

        Select (CASE WHEN A > 1 THEN A ELSE B) as mycol from table1

* `LIKE`

    The ``LIKE`` clause is used to select the strings in a column that matches a pattern::

        SELECT column_name(s) FROM table_name WHERE column LIKE pattern

    In the pattern we support the wildcards ``%`` and ``_``. For example::

        Select A from table1 where B like '%py'


* `GREATEST`

    The ``GREATEST`` clause is used to return the greatest value from a list of columns::

        SELECT GREATEST(col1, col2, ..., colN) from table_name

    For example::

        SELECT GREATEST(A, B, C) from table1

* `With`

    The ``WITH`` clause can be used to name subqueries::

        WITH sub_table AS (SELECT column_name(s) FROM table_name)
        SELECT column_name(s) FROM sub_table

    For example::

        WITH subtable as (Select Max(A) as max_al FROM table1 group by B)
        Select Max(max_val) from subtable


* Aliasing

    SQL aliases are used to give a table, or a column in a table, a temporary name::

        SELECT <COLUMN_NAME> AS <ALIAS>
        FROM <TABLE_NAME>

    For example::

        Select SUM(A) as total from table1


* Operators

    - Bodo SQL currently supports the following arithmetic operators on columns:

        - ``+`` (addition)
        - ``-`` (subtraction)
        - ``*`` (multiplication)
        - ``/`` (true division)

    - Bodo SQL currently supports the following comparision operators on columns:

        - ``=``	(equal to)
        - ``>``	(greater than)
        - ``<``	(less than)
        - ``>=`` (greater than or equal t)o
        - ``<=`` (less than or equal to)
        - ``<>`` (not equal to)

    - Bodo SQL currently supports the following logical operators on columms:

        - ``AND``
        - ``OR``
        - ``NOT``


* Aggregation Functions

    Bodo SQL Currently supports the following Aggregation Functions on all types:

    - COUNT

        Count the number of elements in a column or group.

    In addition, Bodo SQL also supports the following functions on numeric types:

    - AVG

        Compute the mean for a column.

    - MAX

        Compute the max value for a column.

    - MIN

        Compute the min value for a column.

    - STDDEV

        Compute the standard deviation for a column with N - 1 degrees of freedom.

    - STDDEV_SAMP

        Compute the standard deviation for a column with N - 1 degrees of freedom.

    - SUM

        Compute the sum for a column.

    - VARIANCE

        Compute the variance for a column with N - 1 degrees of freedom.

    - VAR_SAMP

        Compute the variance for a column with N - 1 degrees of freedom.


    All aggregate functions have the syntax::

        SELECT AGGREGATE_FUNCTION(<COLUMN_EXPRESSION>)
        FROM <TABLE_NAME>
        GROUP BY <COLUMN_NAMES>


    These functions can be used either in a groupby clause, where they will be computed
    for each group, or by itself on an entire column expression. For example::

        Select AVG(A) from table1 Group By B

        Select Count(Distinct A) from table1


* Timestamp Functions

    Bodo SQL currently supports the following Timestamp functions:

        - DATEDIFF(col1, col2)

            Computes the difference in days between two Timestamp columns

        - STR_TO_DATE(str_col, format_string)

            Converts a string column to a Timestamp columns given a scalar
            format string

        - DATE_ADD(timestamp_col, interval)

            Computes a timestamp column by adding an interval column/scalar
            to a timestamp column

        - DATE_SUB(timestamp_col, interval)

            Computes a timestamp column by subtracting an interval column/scalar
            from a timestamp column

    For example::

        SELECT datediff(A, B) as diff from table1


* String Functions

    Bodo SQL currently supports the following string functions:

        - LOWER(col)

            Converts the contents of the string column to lower case.

        - UPPER(col)

            Converts the contents of the string column to upper case.

    For example::

        SELECT upper(A) as upper_case from table1


Supported Data Types
--------------------
BodoSQL uses Pandas DataFrames to represent SQL tables in memory and converts SQL types
to corresponding Python types which are used by Bodo. Below is a table
mapping SQL types used in BodoSQL to their respective Python types
and Bodo data types.

.. list-table::
  :header-rows: 1

  * - SQL Type(s)
    - Equivalent Python Type
    - Bodo Data Type
  * - ``TINYINT``
    - ``np.int8``
    - ``bodo.int8``
  * - ``SMALLINT``
    - ``np.int16``
    - ``bodo.int16``
  * - ``INT``
    - ``np.int32``
    - ``bodo.int32``
  * - ``BIGINT``
    - ``np.int64``
    - ``bodo.int64``
  * - ``FLOAT``
    - ``np.float32``
    - ``bodo.float32``
  * - ``DECIMAL``, ``DOUBLE``
    - ``np.float64``
    - ``bodo.float64``
  * - ``VARCHAR``, ``CHAR``
    - ``str``
    - ``bodo.string_type``
  * - ``TIMESTAMP``, ``DATE``
    - ``np.datetime64[ns]``
    - ``bodo.datetime64ns``
  * - ``INTERVAL(day-time)``
    - ``np.timedelta64[ns]``
    - ``bodo.timedelta64ns``
  * - ``BOOLEAN``
    - ``np.bool_``
    - ``bodo.bool_``

Nullable and Unsigned Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Although SQL does not explicitly support unsigned types,
by default, BodoSQL maintains the exact types of the existing DataFrames
registered in a `BodoSQLContext`, including unsigned and non-nullable type behavior.
If an operation has the possibility of creating null values or requires
casting data, BodoSQL will convert the input of that operation to a nullable,
signed version of the type.


Supported Literals
------------------

BodoSQL supports the following literal types:
  * :ref:`boolean_literal`
  * :ref:`datetime_literal`
  * :ref:`float_literal`
  * :ref:`integer_literal`
  * :ref:`interval_literal`
  * :ref:`string_literal`


.. _boolean_literal:

Boolean Literal
~~~~~~~~~~~~~~~
**Syntax**::

    TRUE | FALSE

Boolean literals are case insensitive.

.. _datetime_literal:

Datetime Literal
~~~~~~~~~~~~~~~~
**Syntax**::

    DATE 'yyyy-mm-dd' |
    TIMESTAMP 'yyyy-mm-dd' |
    TIMESTAMP 'yyyy-mm-dd HH:mm:ss'

.. _float_literal:

Float Literal
~~~~~~~~~~~~~
**Syntax**::

    [ + | - ] { digit [ ... ] . [ digit [ ... ] ] | . digit [ ... ] }

where digit is any numeral from 0 to 9

.. _integer_literal:

Integer Literal
~~~~~~~~~~~~~~~
**Syntax**::

    [ + | - ] digit [ ... ]

where digit is any numeral from 0 to 9

.. _interval_literal:

Interval Literal
~~~~~~~~~~~~~~~~
**Syntax**::

    INTERVAL integer_literal interval_type

Where integer_literal is a valid integer literal
and interval type is one of::

    DAY[S] |
    HOUR[S] |
    MINUTE[S] |
    SECOND[S]

.. _string_literal:

String Literal
~~~~~~~~~~~~~~
**Syntax**::

    'char [ ... ]'

Where char is a character literal in a Python string.

NULL SEMANTICS
--------------

Bodo SQL converts SQL queries to Pandas code that executes inside Bodo.
As a result, NULL behavior aligns with Pandas and may be slightly different
than other SQL systems. This is currently an area of active development to
ensure compatibility with other SQL systems.

Most operators with a NULL input return NULL. However,
there a couple notable places where Bodo SQL may not match other SQL systems:

    - `GROUP BY` clauses do not produce a NULL group
    - Bodo SQL treats `NaN` the same as NULL
