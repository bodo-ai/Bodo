Common Bodo SQL Errors  {#bodosql_errors}
======================= 

BodoSQL can raise a number of different errors when parsing SQL queries.
This page contains a list of the most commonly encountered parsing errors and their causes.


-  **A binary operation was used on two types for which it is not supported.** 

    ```py
    "Cannot apply 'OP' to arguments of type '<SQL_TYPE_ENUM> OP <SQL_TYPE_ENUM>'"
    ```
    
    This error can be resolved by casting either side of the expression to a common type.

<br>

-  **The format string passed to STR_TO_DATE is not a valid SQL format string.**

    ```py
    "STR_TO_DATE contains an invalid format string"
    ```

    See [`DATE_FORMAT`][date_format] for the list of supported SQL format characters.


<br>

-  **The format string passed to STR_TO_DATE is a valid SQL format string, which contains one or more escape
    characters that BodoSQL does not currently support.**

    ```py
    "STR_TO_DATE contains an invalid escape character (escape char)"
    ```

     For the list of supported SQL format characters, see [`DATE_FORMAT`][date_format].

<br>

-  **The specified column (`COL_NAME`) of one of the pandas DataFrames used to initialize a BodoSQLContext has an unsupported type.**
    

    ```py
    "Pandas column 'COL_NAME' with type PANDAS_TYPE not supported in BodoSQL."
    ```

    For the list of supported pandas types,
    see [here][supported-dataframe-data-types].

<br>

-  **The parser encountered something other than a query at a location where a query was expected.** 
     
     
    ```py
    "Non-query expression encountered in illegal context"
    ```

    See here for the syntax of a [select clause][select].

<br>

-  **The table name specified in a SQL query doesn't match a table name registered in the BodoSQLContext.** 
    
    
    ```py
    "Object 'tablename' not found"
    ```

    Generally, this is caused by misnaming a table when initializing the BodoSqlContext, or misnaming a table in the query itself.

<br>

-  **The query attempted to select a column from one or more tables, and the column wasn't present in any of them.**


    ```py
    "Column 'COL_NAME' not found in any table"
    ```

     Generally, this is caused by misnaming a column while initializing the BodoSQLContext, or misnaming a column in the query itself.``

<br>

-  **The query attempted to select a column for two or more tables, and the column was present in multiple tables.**

     
    ```py
    "Column 'COL_NAME' is ambiguous"
    ```
  
<br>

-  **The types of arguments supplied to the function don't match the types supported by the function.** 
     
     
    ```py
    "Cannot apply 'FN_NAME' to arguments of type 'FN_NAME(<ARG1_SQL_TYPE>, <ARG2_SQL_TYPE>, ...)'. Supported form(s): 'FN_NAME(<ARG1_SQL_TYPE>, <ARG2_SQL_TYPE>, ...)'"
    ```

    Generally, this can be resolved by the specifying the origin table like so:

    ``Select A from table1, table2`` → ``Select table1.A from table1, table2``

    This can be resolved by explicitly casting the problematic argument(s) to the appropriate type.
    For the list of functions and the argument types they support, [see here][supported-operations].

<br>

-  **Either BodoSQL doesn't support the function or an incorrect number of arguments was supplied.**
     
     
    ```py
    "No match found for function signature FN_NAME(<ARG1_SQL_TYPE>, <ARG2_SQL_TYPE>, ...)"
    ```

    In both cases, the list of functions which we support, and their calling conventions can be found [here][supported-operations].

<br>

- **A Window function that does not support windows with a `ROWS_BETWEEN` clause was called over a window containing a `ROWS_BETWEEN` clause.**
    
   
    ```py
    "ROW/RANGE not allowed with RANK, DENSE_RANK or ROW_NUMBER functions"
    ```

     In addition to the RANK, DENSE_RANK, or ROW_NUMBER functions listed in the error message, LEAD and LAG also have this requirement.
    The list of window aggregations we support, and their calling syntax can be found [here][window-functions].

<br>

-  **BodoSQL was unable to parse your SQL because the query contained unsupported syntax.**
    
    
    ```py
    "Encountered "KEYWORD" at line X, column Y. Was expecting one of: ..."
    ```

     There are a variety of reasons this could occur, but here are some of the common ones:
    
    * A typo in one of the query words, for example ``groupby`` instead of ``group by``. In this situation ``line X, column Y`` should point you to the first typo.
    * All the components are legal SQL keywords, but they are used in an incorrect order. Please refer to our support syntax to check for legal constructions. If you believe your query should be supported [please file an issue](https://github.com/Bodo-inc/Feedback).
    * Trying to use double-quotes for a string literal (i.e. ```py"example"
    ``` instead of ``'example'``)
    * Unclosed parenthesis or trailing commas

<br>

-  **A parameter was not properly registered in the BodoSQLContext.**

     
    ```py
    "SQL query contains a unregistered parameter: '@param_name'"
    ```

    This is often caused by failing to pass the parameter to `BodoSQLContext.sql()` or using an incorrect name in either the query or the registration. For more information on named parameters, see [here][bodosql_named_params].

<br>

- **A non-dataframe value was used to initialize a BodoSQLContext.**


    ```py
    "BodoSQLContext(): 'table' values must be DataFrames"
    ```

    The dictionary used to initialize a BodoSQLContext must map string table names to pandas DataFrames.
