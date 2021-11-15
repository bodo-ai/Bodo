.. _bodosql_errors:

Common Bodo SQL Errors
=======================

BodoSQL can raise a number of different errors when parsing SQL queries. Below is a list of the most commonly encountered parsing errors and their causes.


           - ``"Cannot apply 'OP' to arguments of type '<SQL_TYPE_ENUM> OP <SQL_TYPE_ENUM>'"``

    A binary operation was used on two types for which it is not supported. This error can be resolved by casting either side of the expression to a common type.



           - ``"STR_TO_DATE contains an invalid format string"``

    The format string passed to STR_TO_DATE is not a valid SQL format string.
    See :ref:`DATE_FORMAT <date_formating_characters>` for the list of supported SQL format characters.

           - ``"STR_TO_DATE contains an invalid escape character (escape char)"``

    The format string passed to STR_TO_DATE is a valid SQL format string, which contains one or more escape
    characters that BodoSQL does not currently support. For the list of supported SQL format characters, see :ref:`DATE_FORMAT <date_formating_characters>`.

           - ``"Pandas column 'COL_NAME' with type PANDAS_TYPE not supported in BodoSQL."``

    The specified column (`COL_NAME`) of one of the pandas DataFrames used to initialize the BodoSQLContext has an unsupported type. For the list of supported pandas types,
    see :ref:`here <supported_dataframe_data_types>`.

           - ``"BodoSQLContext(): 'table' values must be DataFrames"``

    A non-dataframe value was used to initialize a BodoSQLContext. The dictionary used to initialize a BodoSQLContext must map string table names to pandas DataFrames.

           - ``"Non-query expression encountered in illegal context"``

    The parser encountered something other than a query at a location where a query was expected. :ref:`See here for the syntax of a select clause <select_clause>`.

           - ``"Object 'tablename' not found"``

    The table name specified in a SQL query doesn't match a table name registered in the BodoSQLContext. Generally, this is caused by misnaming a table when initializing the BodoSqlContext, or misnaming a table in the query itself.

           - ``"Column 'COL_NAME' not found in any table"``

    The query attempted to select a column from one or more tables, and the column wasn't present in any of them. Generally, this is caused by misnaming a column while initializing the BodoSQLContext, or misnaming a column in the query itself.``

           - ``"Column 'COL_NAME' is ambiguous"``

    The query attempted to select a column for two or more tables, and the column was present in multiple tables.


           - ``"Cannot apply 'FN_NAME' to arguments of type 'FN_NAME(<ARG1_SQL_TYPE>, <ARG2_SQL_TYPE>, ...)'. Supported form(s): 'FN_NAME(<ARG1_SQL_TYPE>, <ARG2_SQL_TYPE>, ...)'"``

    Generally, this can be resolved by the specifying the origin table like so:


        ``Select A from table1, table2`` → ``Select table1.A from table1, table2``


    The types of arguments supplied to the function don't match the types supported by the function. This can be resolved by explicitly casting the problematic argument(s) to the appropriate type.
    For the list of functions and the argument types they support, :ref:`see here <bodosql_fns_start>`.

           - ``"No match found for function signature FN_NAME(<ARG1_SQL_TYPE>, <ARG2_SQL_TYPE>, ...)"``

    Either BodoSQL doesn't support the function or an incorrect number of arguments was supplied.
    In both cases, the list of functions which we support, and their calling conventions can be found :ref:`here <bodosql_fns_start>`.

           - ``"ROW/RANGE not allowed with RANK, DENSE_RANK or ROW_NUMBER functions"``

    A Window function that does not support windows with a `ROWS_BETWEEN` clause was called over a window containing a `ROWS_BETWEEN` clause. In addition to the RANK, DENSE_RANK, or ROW_NUMBER functions listed in the error message, LEAD and LAG also have this requirement.
    The list of window aggregations we support, and their calling syntax can be found :ref:`here <window_fns>`.

           - ``"Encountered "KEYWORD" at line X, column Y. Was expecting one of: ..."``

    BodoSQL was unable to parse your SQL because the query contained unsupported syntax. There are a variety of reasons this could occur, but here are some of the common ones:

        * A typo in one of the query words, for example ``groupby`` instead of ``group by``. In this situation ``line X, column Y`` should point you to the first typo.
        * All the components are legal SQL keywords, but they are used in an incorrect order. Please refer to our support syntax to check for legal constructions. If you believe your query should be supported `please file an issue <https://github.com/Bodo-inc/Feedback>`_.
        * Trying to use double-quotes for a string literal (i.e. ``"example"`` instead of ``'example'``
        * Unclosed parenthesis or trailing commas

           - ``"SQL query contains a unregistered parameter: '@param_name'"``

    The parameter 'param_name' was not is not properly registered in the BodoSQLContext. This is often caused by failing to pass the parameter to BodoSQLContext.sql() or using an incorrect name in either the query or the registration. For more information on named parameters, see :ref:`here <bodosql_named_params>`.
