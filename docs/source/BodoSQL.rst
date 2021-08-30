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
Bodo SQL is currently in Beta. Install it using::

    conda install bodosql -c bodo.ai -c conda-forge

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
Bodo SQL ignores casing of keywords, and column and table names, except for the final output column name.
Therefore, ``select a from table1`` is treated the same as ``SELECT A FROM Table1``, except for the names of
the final output columns (``a`` vs ``A``).

* `SELECT`

    The ``SELECT`` statement is used to select data in the form of columns. The data returned from Bodo SQL is stored in a dataframe. Example usage::

        SELECT <COLUMN_NAMES> FROM <TABLE_NAME>

    For Example::

        SELECT A FROM table1

    The ``SELECT DISTINCT`` statement is used to return only distinct (different) values::

        SELECT DISTINCT <COLUMN_NAMES> FROM <TABLE_NAME>

    ``DISTINCT`` can be used in a SELECT statement or inside an aggregate function. For example::

        SELECT DISTINCT A FROM table1

        SELECT COUNT DISTINCT A FROM table1


* `WHERE`

    The ``WHERE`` clause on columns can be used to filter records that satisfy specific conditions::

        SELECT <COLUMN_NAMES> FROM <TABLE_NAME> WHERE <CONDITION>

    For Example::

        SELECT A FROM table1 WHERE B > 4


* `ORDER BY`

    The ``ORDER BY`` keyword sorts the resulting dataframe in ascending or descending order. By default, it sorts the records in ascending order.
    For descending order, the ``DESC`` keyword can be used::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        ORDER BY <ORDERED_COLUMN_NAMES> ASC|DESC

    For Example::

        SELECT A, B FROM table1 ORDER BY B, A DESC


* `LIMIT`

    Bodo SQL supports the ``LIMIT`` keyword to select a limited number of rows.
    This keyword can optionally include an offset::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <CONDITION>
        LIMIT <LIMIT_NUMBER> OFFSET <OFFSET_NUMBER>

    For Example::

        SELECT A FROM table1 LIMIT 5

        SELECT B FROM table2 LIMIT 8 OFFSET 3

    Specifying a limit and offset can be also be written as::

        LIMIT <OFFSET_NUMBER>, <LIMIT_NUMBER>

    For Example::

        SELECT B FROM table2 LIMIT 3, 8


* [NOT] `IN`

    The ``IN`` determines if a value can be chosen a list of options.
    Currently we support lists of literals or columns with matching types.

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> IN (<val1>, <val2>, ... <valN>)

    For example::

        SELECT A FROM table1 WHERE A IN (5, 10, 15, 20, 25)


* [NOT] `BETWEEN`

    The ``BETWEEN`` operator selects values within a given range. The values can be numbers, text, or datetimes.
    The ``BETWEEN`` operator is inclusive: begin and end values are included::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <COLUMN_NAME> BETWEEN <VALUE1> AND <VALUE2>

    For example::

        SELECT A FROM table1 WHERE A BETWEEN 10 AND 100


* `CAST`

    THE ``CAST`` operator converts an input from one type to another. In many cases
    casts are created implicitly, but this operator can be used to force a type
    conversion.

    The following casts are currently supported. Please refer to :ref:`supported_dataframe_data_types`
    for the Python types for each type keyword:

        - VARCHAR → VARCHAR

        - VARCHAR → TINYINT/SMALLINT/INTERGER/BIGINT

        - VARCHAR → FLOAT/DOUBLE

        - VARCHAR → DECIMAL

            - Equivalent to DOUBLE. This may change in the future

        - VARCHAR → TIMESTAMP

        - VARCHAR → DATE

            - Truncates to date but is still Timestamp type. This may change in the future.

        - TINYINT/SMALLINT/INTERGER/BIGINT → VARCHAR

        - TINYINT/SMALLINT/INTERGER/BIGINT → TINYINT/SMALLINT/INTERGER/BIGINT

        - TINYINT/SMALLINT/INTERGER/BIGINT → FLOAT/DOUBLE

        - TINYINT/SMALLINT/INTERGER/BIGINT → DECIMAL

            - Equivalent to DOUBLE. This may change in the future

        - TINYINT/SMALLINT/INTERGER/BIGINT → TIMESTAMP

        - FLOAT/DOUBLE → VARCHAR

        - FLOAT/DOUBLE → TINYINT/SMALLINT/INTERGER/BIGINT

        - FLOAT/DOUBLE → FLOAT/DOUBLE

        - FLOAT/DOUBLE → DECIMAL

            - Equivalent to DOUBLE. This may change in the future

        - TIMESTAMP → VARCHAR

        - TIMESTAMP → TINYINT/SMALLINT/INTERGER/BIGINT

        - TIMESTAMP → TIMESTAMP

        - TIMESTAMP → DATE

            - Truncates to date but is still Timestamp type. This may change in the future.

    *Note*: CAST correctness can often not be determined at compile time. Users are responsible
        for ensuring that conversion is possible (e.g. ``CAST(str_col as INTEGER)``).


* `JOIN`

    A ``JOIN`` clause is used to combine rows from two or more tables, based on a related column between them::

      SELECT <COLUMN_NAMES>
        FROM <LEFT_TABLE_NAME>
        <JOIN_TYPE> <RIGHT_TABLE_NAME>
        ON <LEFT_TABLE_COLUMN_NAME> = <RIGHT_TABLE_COLUMN_NAME>


    For example::

        SELECT table1.A, table1.B FROM table1 JOIN table2 on table1.A = table2.C

    Here are the different types of the joins in SQL:

    - ``(INNER) JOIN``: returns records that have matching values in both tables
    - ``LEFT (OUTER) JOIN``: returns all records from the left table, and the matched records from the right table
    - ``RIGHT (OUTER) JOIN``: returns all records from the right table, and the matched records from the left table
    - ``FULL (OUTER) JOIN``: returns all records when there is a match in either left or right table

    Bodo SQL currently supports inner join on all conditions, but all outer joins are only supported on an
    equality between columns.

* `UNION`

    The UNION operator is used to combine the result-set of two SELECT statements::

        SELECT <COLUMN_NAMES> FROM <TABLE1>
        UNION
        SELECT <COLUMN_NAMES> FROM <TABLE2>

    Each SELECT statement within the UNION caluse must have the same number of columns. The columns must also have similar
    data types. The output of the UNION is the set of rows which are present in either of the input SELECT statements.

    The UNION operator selects only the distinct values from the inputs by default. To allow duplicate values, use UNION ALL::

        SELECT <COLUMN_NAMES> FROM <TABLE1>
        UNION ALL
        SELECT <COLUMN_NAMES> FROM <TABLE2>


* `INTERSECT`

    The INTERSECT operator is used to calculate the intersection of two SELECT statements::

        SELECT <COLUMN_NAMES> FROM <TABLE1>
        INTERSECT
        SELECT <COLUMN_NAMES> FROM <TABLE2>

    Each SELECT statement within the INTERSECT clause must have the same number of columns.
    The columns must also have similar data types. The output of the INTERSECT is the set of rows which are present in
    both of the input SELECT statements. The INTERSECT operator selects only the distinct values from the inputs.


* `GROUP BY`
    The ``GROUP BY`` statement groups rows that have the same values into summary rows, like "find the number of customers in each country".
    The ``GROUP BY`` statement is often used with aggregate functions to group the result-set by one or more columns::

        SELECT <COLUMN_NAMES>
        FROM <TABLE_NAME>
        WHERE <CONDITION>
        GROUP BY <COLUMN_NAMES>
        ORDER BY <COLUMN_NAMES>

    For example::

        SELECT MAX(A) FROM table1 GROUP BY B

    ``GROUP BY`` statements also referring to columns by alias or column number::

        SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val
        SELECT MAX(A), B FROM table1 GROUP BY 2


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

        SELECT MAX(A) FROM table1 GROUP BY B HAVING C < 0

    ``HAVING`` statements also referring to columns by aliases used in the ``GROUP BY``::

        SELECT MAX(A), B - 1 as val FROM table1 GROUP BY val HAVING val > 5

* `CASE`

    The ``CASE`` statement goes through conditions and returns a value when the first condition is met::

        SELECT CASE WHEN cond1 THEN value1 WHEN cond2 THEN value2 ... ELSE valueN END

    For example::

        SELECT (CASE WHEN A > 1 THEN A ELSE B END) as mycol FROM table1

    If the types of the possible return values are different, BodoSQL will attempt to cast them all to a common type,
    which is currently undefined behavior. The last else clause can optionally be excluded, in which case, the
    CASE statement will return null if none of the conditions are met. For example::

        SELECT (CASE WHEN A < 0 THEN 0 END) as mycol FROM table1

    is equivalent to::

        SELECT (CASE WHEN A < 0 THEN 0 ELSE NULL END) as mycol FROM table1


* `LIKE`

    The ``LIKE`` clause is used to filter the strings in a column to those that match a pattern::

        SELECT column_name(s) FROM table_name WHERE column LIKE pattern

    In the pattern we support the wildcards ``%`` and ``_``. For example::

        SELECT A FROM table1 WHERE B LIKE '%py'


* `GREATEST`

    The ``GREATEST`` clause is used to return the largest value from a list of columns::

        SELECT GREATEST(col1, col2, ..., colN) FROM table_name

    For example::

        SELECT GREATEST(A, B, C) FROM table1


* `LEAST`

    The ``LEAST`` clause is used to return the smallest value from a list of columns::

        SELECT LEAST(col1, col2, ..., colN) FROM table_name

    For example::

        SELECT LEAST(A, B, C) FROM table1

* `With`

    The ``WITH`` clause can be used to name subqueries::

        WITH sub_table AS (SELECT column_name(s) FROM table_name)
        SELECT column_name(s) FROM sub_table

    For example::

        WITH subtable as (SELECT MAX(A) as max_al FROM table1 GROUP BY B)
        SELECT MAX(max_val) FROM subtable


* Aliasing

    SQL aliases are used to give a table, or a column in a table, a temporary name::

        SELECT <COLUMN_NAME> AS <ALIAS>
        FROM <TABLE_NAME>

    For example::

        Select SUM(A) as total FROM table1

    We strongly recommend using aliases for the final outputs of any queries to ensure
    all column names are predictable.


* Operators

    - Bodo SQL currently supports the following arithmetic operators:

        - ``+`` (addition)
        - ``-`` (subtraction)
        - ``*`` (multiplication)
        - ``/`` (true division)
        - ``%`` (modulo)

    - Bodo SQL currently supports the following comparision operators:

        - ``=``	(equal to)
        - ``>``	(greater than)
        - ``<``	(less than)
        - ``>=`` (greater than or equal to)
        - ``<=`` (less than or equal to)
        - ``<>`` (not equal to)
        - ``!=`` (not equal to)
        - ``<=>`` (equal to or both inputs are null)

    - Bodo SQL currently supports the following logical operators:

        - ``AND``
        - ``OR``
        - ``NOT``

    - Bodo SQL currently supports the following string operators:

        - ``||`` (string concatination)



* Numeric Functions

    Except where otherwise specified, the inputs to each of these functions can be any numeric
    type, column or scalar. Here is an example using MOD::

        SELECT MOD(12.2, A) FROM table1

    Bodo SQL Currently supports the following Numeric Functions:

    - ABS(n)

        Returns the absolute value of n

    - COS(n)

        Calculates the Cosine of n

    - SIN(n)

        Calculates the Sine of n

    - TAN(n)

        Calculates the Tangent of n

    - ACOS(n)

        Calculates the Arccosine of n

    - ASIN(n)

        Calculates the Arcsine of n

    - ATAN(n)

        Calculates the Arctangent of n

    - ATAN2(A, B)

        Calculates the Arctangent of A divided by B

    - COTAN(X)

        Calculates the Cotangent of X

    - CEIL(X)
        Converts X to an integer, rounding towards positive infinity

    - CEILING(X)

        Equivalent to CEIL

    - FLOOR(X)

        Converts X to an integer, rounding towards negative infinity

    - DEGREES(X)

        Converts a value in radians to the corresponding value in degrees

    - RADIANS(X)

        Converts a value in radians to the corresponding value in degrees

    - LOG10(X)

        Computes Log base 10 of x. Returns NaN for negative inputs, and -inf for 0 inputs.

    - LOG(X)

        Equivalent to LOG10(x)

    - LOG10(X)

        Computes Log base 2 of x. Returns NaN for negative inputs, and -inf for 0 inputs.

    - LN(X)

        Computes the natural log of x. Returns NaN for negative inputs, and -inf for 0 inputs.

    - MOD(A,B)

        Computes A modulo B.

    - CONV(X, current_base, new_base)

        CONV takes a string representation of an integer value, it's current_base, and the base to convert that argument to.
        CONV returns a new string, that represents the value in the new base. CONV is only supported for converting to/from
        base 2, 8, 10, and 16.

        For example::

            CONV('10', 10, 2) ==> '1010'
            CONV('10', 2, 10) ==> '2'
            CONV('FA', 16, 10) ==> '250'


    - SQRT(X)

        Computes the square root of x. Returns NaN for negative inputs, and -inf for 0 inputs.

    - PI()

        Returns the value of PI

    - POW(A, B), POWER(A, B)

        Returns A to the power of B. Returns NaN if A is negative, and B is a float. POW(0,0) is 1

    - EXP(X)

        Returns e to the power of X

    - SIGN(X)

        Returns 1 if X > 0, -1 if X < 0, and 0 if X = 0

    - ROUND(X, num_decimal_places)

        Rounds X to the specified number of decimal places

    - TRUNCATE(X, num_decimal_places)

        Equivalent to ROUND(X, num_decimal_places)


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

    - STDDEV_POP

        Compute the standard deviation for a column with N degrees of freedom.

    - SUM

        Compute the sum for a column.

    - VARIANCE

        Compute the variance for a column with N - 1 degrees of freedom.

    - VAR_SAMP

        Compute the variance for a column with N - 1 degrees of freedom.

    - VAR_POP

        Compute the variance for a column with N degrees of freedom.


    All aggregate functions have the syntax::

        SELECT AGGREGATE_FUNCTION(<COLUMN_EXPRESSION>)
        FROM <TABLE_NAME>
        GROUP BY <COLUMN_NAMES>


    These functions can be used either in a groupby clause, where they will be computed
    for each group, or by itself on an entire column expression. For example::

        SELECT AVG(A) FROM table1 GROUP BY B

        SELECT COUNT(Distinct A) FROM table1


* Timestamp Functions

    Bodo SQL currently supports the following Timestamp functions:

        - DATEDIFF(timestamp_val1, timestamp_val2)

            Computes the difference in days between two Timestamp values

        - STR_TO_DATE(str_val, literal_format_string)

            Converts a string value to a Timestamp value given a literal
            format string. If a year, month, and day value is not specified,
            they default to 1900, 01, and 01 respectively. Will throw a runtime error
            if the string cannot be parsed into the expected values. See DATE_FORMAT for
            Recognized formatting characters.

        For example::

                STR_TO_DATE('2020 01 12', '%Y %m %d') ==> Timestamp '2020-01-12'
                STR_TO_DATE('01 12', '%m %d') ==> Timestamp '1900-01-12'
                STR_TO_DATE('hello world', '%Y %m %d') ==> RUNTIME ERROR

        - DATE_FORMAT(timestamp_val, literal_format_string)

            Converts a timestamp value to a String value given a scalar
            format string.

            Recognized formatting character:
                - ``%i`` Minutes, zero padded (00 to 59)
                - ``%M`` Full month name (January to December)
                - ``%r`` Time in format in the format (hh:mm:ss AM/PM)
                - ``%s`` Seconds, zero padded (00 to 59)
                - ``%T`` Time in format in the format (hh:mm:ss)
                - ``%T`` Time in format in the format (hh:mm:ss)
                - ``%u`` week of year, where monday is the first day of the week (00 to 53)
                - ``%a`` Abbreviated weekday name (sun-sat)
                - ``%b`` Abbreviated month name (jan-dec)
                - ``%f`` Microseconds, left padded with 0's, (000000 to 999999)
                - ``%H`` Hour, zero padded (00 to 23)
                - ``%j`` Day Of Year, left padded with 0's (001 to 366)
                - ``%m`` Month number (00 to 12)
                - ``%p`` AM or PM, depending on the time of day
                - ``%d`` Day of month, zero padded (01 to 31)
                - ``%Y`` Year as a 4 digit value
                - ``%y`` Year as a 2 digit value, zero padded (00 to 99)
                - ``%U`` Week of year where sunday is the first day of the week (00 to 53)
                - ``%S`` Seconds, zero padded (00 to 59)

            For example::

                DATE_FORMAT(Timestamp '2020-01-12', '%Y %m %d') ==> '2020 01 12'
                DATE_FORMAT(Timestamp '2020-01-12 13:39:12', 'The time was %T %p. It was a %u') ==> 'The time was 13:39:12 PM. It was a Sunday'


        - DATE_ADD(timestamp_val, interval)

            Computes a timestamp column by adding an interval column/scalar
            to a timestamp value

        - DATE_SUB(timestamp_val, interval)

            Computes a timestamp column by subtracting an interval column/scalar
            to a timestamp value

        - NOW()

            Computes a timestamp equal to the current system time

        - LOCALTIMESTAMP()

            Equivalent to NOW

        - CURDATE()

            Computes a timestamp equal to the current system time, excluding the time information

        - CURRENT_DATE()

            Equivalent to CURDATE

        - EXTRACT(TimeUnit from timestamp_val)

            Extracts the specified TimeUnit from the supplied date.

            allowed TimeUnits are:
                - MICROSECOND
                - SECOND
                - MINUTE
                - HOUR
                - DAY (Day of Month)
                - DOY (Day of Year)
                - DOW (Day of week)
                - WEEK
                - MONTH
                - QUARTER
                - YEAR

            TimeUnits are not case sensitive.

        - MICROSECOND(timestamp_val),

            Equivalent to EXTRACT(MICROSECOND from timestamp_val)

        - SECOND(timestamp_val)

            Equivalent to EXTRACT(SECOND from timestamp_val)

        - MINUTE(timestamp_val)

            Equivalent to EXTRACT(MINUTE from timestamp_val)

        - HOUR(timestamp_val)

            Equivalent to EXTRACT(HOUR from timestamp_val)

        - WEEK(timestamp_val)

            Equivalent to EXTRACT(WEEK from timestamp_val)

        - WEEKOFYEAR(timestamp_val)

            Equivalent to EXTRACT(WEEK from timestamp_val)

        - MONTH(timestamp_val)

            Equivalent to EXTRACT(MONTH from timestamp_val)

        - QUARTER(timestamp_val)

            Equivalent to EXTRACT(QUARTER from timestamp_val)

        - YEAR(timestamp_val)

            Equivalent to EXTRACT(YEAR from timestamp_val)

        - MAKEDATE(integer_years_val, integer_days_val)

            Computes a timestamp value that is the specified number of days after the specified year.

        - DAYNAME(timestamp_val)

            Computes the string name of the day of the timestamp value.

        - MONTHNAME(timestamp_val)

            Computes the string name of the month of the timestamp value.

        - TO_DAYS(timestamp_val)

            Computes the difference in days between the input timestamp, and year 0 of the Gregorian calendar

        - TO_SECONDS(timestamp_val)

            Computes the number of seconds since year 0 of the Gregorian calendar

        - FROM_DAYS(n)

            Returns a timestamp values that is n days after year 0 of the Gregorian calendar

        - UNIX_TIMESTAMP()

            Computes the number of seconds since the unix epoch

        - FROM_UNIXTIME(n)

            Returns a Timestamp value that is n seconds after the unix epoch

        - ADDDATE(timestamp_val, interval)

            Same as DATE_ADD

        - SUBDATE(timestamp_val, interval)

            Same as DATE_SUB

        - TIMESTAMPDIFF(unit, timestamp_val1, timestamp_val2)

            Returns timestamp_val1 - timestamp_val2 rounded down
            to the provided unit.

        - WEEKDAY(timestamp_val)

            Returns the weekday number for timestamp_val.
            Note: Monday = 0, Sunday=6


        - YEARWEEK(timestamp_val)

            Returns the year and week number for the provided timestamp_val
            concatenated as a single number. For example::

                YEARWEEK(TIMESTAMP '2021-08-30::00:00:00')
                202135

        - LAST_DAY(timestamp_val)

            Given a timestamp value, returns a timestamp value that is the
            last day in the same month as timestamp_val.



* String Functions

    Bodo SQL currently supports the following string functions:

        - LOWER(str)

            Converts the string scalar/column to lower case.

        - LCASE(str)

            Same as LOWER.

        - UPPER(str)

            Converts the string scalar/column to upper case.

        - UCASE(str)

            Same as UPPER.

        - CONCAT(str_0, str_1, ...)

            Concatinates the strings together. Requires at least two arguments.

        - CONCAT_WS(str_separator, str_0, str_1, ...)

            Concatinates the strings together, with the specified separator. Requires at least three arguments

        - SUBSTRING(str, start_index, len)

            Takes a substring of the specified string, starting at the specified index, of the specified length.
            Start_index = 1 specfies the first character of the string, start_index = -1 specfies the last
            character of the string. Start_index = 0 causes the function to return empty string. If start_index is positive and greater then the length of the string, returns
            an empty string. If start_index is negative, and has an absolute value greater then the length of the string,
            the behavior is equivalent to start_index = 1.

            For example::

                SUBSTRING('hello world', 1, 5) ==> 'hello'
                SUBSTRING('hello world', -5, 7) ==> 'world'
                SUBSTRING('hello world', -20, 8) ==> 'hello wo'
                SUBSTRING('hello world', 0, 10) ==> ''


        - MID(str, start_index, len)

            Equivalent to SUBSTRING

        - SUBSTR(str, start_index, len)

            Equivalent to SUBSTRING

        - LEFT(str, n)

            Takes a substring of the specified string consisting of the leftmost n characters

        - RIGHT(str, n)

            Takes a substring of the specified string consisting of the rightmost n characters

        - REPEAT(str, len)

            Extends the specified string to the specified length by repeating the string. Will truncate the string
            If the string's length is less then the len argument

            For example::

                REPEAT('abc', 7) ==> 'abcabca'
                REPEAT('hello world', 5) ==> 'hello'

        - STRCMP(str1, str2)

            Compares the two strings lexographically.
            If str1 > str2, return 1. If str1 < str2, returns -1. If str1 = str2, returns 0.

        - REVERSE(str)

            Returns the reversed string.

        - ORD(str)

            Returns the integer value of the unicode representation of the first charecter of the input string.
            returns 0 when passed the empty string

        - CHAR(int)

            Returns the charecter of the corresponding unicode value.
            Currently only supported for ASCII charecters (0 to 127, inclusive)

        - SPACE(int)

            Returns a string containing the specified number of spaces.

        - LTRIM(str)

            returns the input string, will all spaces removed from the left of the string

        - RTRIM(str)

            returns the input string, will all spaces removed from the right of the string

        - TRIM(str)

            returns the input string, will all spaces removed from the left and right of the string

        - SUBSTRING_INDEX(str, delimiter_str, n)

            Returns a substring of the input string, which contains all characters that occur before
            n occurances of the delimiter string. if n is negative, it will return all characters
            that occur after the last n occurances of the delimiter string. If num_occurances is 0,
            it will return the empty string

            For example::

                SUBSTRING_INDEX('1,2,3,4,5', ',', 2) ==> '1,2'
                SUBSTRING_INDEX('1,2,3,4,5', ',', -2) ==> '4,5'
                SUBSTRING_INDEX('1,2,3,4,5', ',', 0) ==> ''

        - LPAD(string, len, padstring)

            Extends the input string to the specified length, by appending copies of the padstring to the
            left of the string. If the input string's length is less then the len argument, it will truncate
            the input string.

            For example::

                LPAD('hello', 10, 'abc') ==> 'abcabhello'
                LPAD('hello', 1, 'abc') ==> 'h'

        - RPAD(string, len, padstring)

            Extends the input string to the specified length, by appending copies of the padstring to the
            right of the string. If the input string's length is less then the len argument, it will truncate
            the input string.

            For example::

                RPAD('hello', 10, 'abc') ==> 'helloabcab'
                RPAD('hello', 1, 'abc') ==> 'h'

        - REPLACE(base_string, substring_to_remove, string_to_substitute)

            Replaces all occurances of the specified substring with the substitute string.

            For example::

                REPLACE('hello world', 'hello' 'hi') ==> 'hi world'

        - LENGTH(string)

            Returns the number of characters in the given string.


* Control flow Functions

    - IF(Cond, TrueValue, FalseValue)

        Returns TrueValue if cond is True, and FalseValue if cond is false. Loigcally equivalent to::

            CASE WHEN Cond THEN TrueValue ELSE FalseValue END

    - IFNULL(Arg0, Arg1)

        Returns Arg1 if Arg0 is null, and otherwise returns Arg1. If Arguments do not have the same
        type, Bodo SQL will attempt to cast them all to a common type, which is currently undefined behavior.

    - NVL(Arg0, Arg1)

        Equivalent to IFNULL

    - NULLIF(Arg0, Arg1)

        Returns null if the Arg0 evaluates to true, and otherwise returns Arg1

    - COALESCE(A, B, C, ...)

        Returns the first non NULL argument, or NULL if no non NULL argument is found. Requires at least
        two arguments. If Arguments do noth have the same type, Bodo SQL will attempt to cast them to a
        common datatype, which is currently undefined behavior.


.. _supported_dataframe_data_types:

Supported DataFrame Data Types
------------------------------
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

In addition we also have limited suport for YEAR[S] and MONTH[S].
These literals cannot be stored in columns and currently are only
supported for operations involving add and sub.

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

    - Bodo SQL treats `NaN` the same as NULL
    - Is (NOT) False and Is (NOT) True return NULL when used on a null expression
    - AND will return NULL if any of the inputs is NULL


BodoSQL Caching & Parameterized Queries
---------------------------------------

BodoSQL can reuse Bodo caching to avoid recompilation when used inside a JIT function.
BodoSQL caching works the same as Bodo, so for example::

    @bodo.jit(cache=True)
    def f(filename):
        df1 = pd.read_parquet(filename)
        bc = bodosql.BodoSQLContext({"table1": df1})
        df2 = bc.sql("SELECT A FROM table1 WHERE B > 4")
        print(df2.A.sum())

This will avoid recompilation so long as the DataFrame scheme stored in ``filename``
has the same schema and the code does not change.

To enable caching for queries with scalar parameters that you may want to adjust
between runs, we introduce a feature called parameterized queries. In a parameterized
query, the SQL query replaces a constant/scalar value with a variable,
which we call a named parameter. In addition, the query is passed a dictionary
of parameters which maps each name to a corresponding Python variable.

For example, if in the above SQL query we wanted to replace 4 with other integers,
we could rewrite our query as::

    bc.sql("SELECT A FROM table1 WHERE B > @var", {"var": python_var})

Now anywhere that ``@var`` is used, the value of python_var at runtime will be used
instead. This can be used in caching, because python_var can be provided as an argument
to the JIT function itself, thus enabling changing the filter without recompiling. The
full example looks like this::

    @bodo.jit(cache=True)
    def f(filename, python_var):
        df1 = pd.read_parquet(filename)
        bc = bodosql.BodoSQLContext({"table1": df1})
        df2 = bc.sql("SELECT A FROM table1 WHERE B > @var", {"var": python_var})
        print(df2.A.sum())


Named parameters cannot be used in places that require a constant value to generate
the correct implementation (e.g. TimeUnit in EXTRACT).
