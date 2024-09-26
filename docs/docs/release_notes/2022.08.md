Bodo 2022.8 Release (Date: 08/31/2022) {#August_2022}
========================================

## New Features and Improvements

Compilation / Performance improvements:

- BodoSQL generated plans are now more optimized to reduce runtime, compile time, and memory usage.
- Performance improvements to pivot_table by reducing the amount of data being shuffled.
- BodoSQL `CASE` statements are now faster to compile.

I/O:

- Bodo now uses a new optimized connector to write to Snowflake efficiently in parallel (with standard `DataFrame.to_sql()` syntax).
- Support for reading strings columns with dictionary encoding when fetching data from Snowflake.
- Bodo is now upgraded to use Arrow 8.
- Bodo can avoid loading any columns with parquet if only the length needs to be computed.


Iceberg:

- Support for limit pushdown with data read from Iceberg.


Pandas coverage:

- Added support for dictionary-encoded string arrays (that have reduced memory usage and execution time) with `pandas.concat`
- Support for `groupby.sum()` with boolean columns.
- Support for `MultiIndex.nbytes`
- Support for `Series.str.index`
- Support for `Series.str.rindex`


BodoSQL:

- Update the default null ordering with `ORDER BY` (nulls first with ASC, nulls last with DESC).

- Updates aggregation without a `GROUP BY` to return a replicated result.

- Improved runtime performance when computing a `SUM` inside a window function.

- Added support for the following column functions

    - `ACOSH`
    - `ASINH`
    - `ATANH`
    - `BITAND`
    - `BITOR`
    - `BITXOR`
    - `BITNOT`
    - `BITSHIFTLEFT`
    - `BITSHIFTRIGHT`
    - `BOOLAND`
    - `BOOLNOT`
    - `BOOLOR`
    - `BOOLXOR`
    - `CBRT`
    - `COSH`
    - `DATEADD`
    - `DECODE`
    - `DIV0`
    - `EDITDISTANCE`
    - `EQUAL_NULL`
    - `FACTORIAL`
    - `GETBIT`
    - `HAVERSINE`
    - `INITCAP`
    - `REGEXP`
    - `REGEXP_COUNT`
    - `REGEXP_INSTR`
    - `REGEXP_LIKE`
    - `REGEXP_REPLACE`
    - `REGEXP_SUBSTR`
    - `REGR_VALX`
    - `REGR_VALY`
    - `RLIKE`
    - `SINH`
    - `SPLIT_PART`
    - `SQUARE`
    - `STRTOK`
    - `TANH`
    - `TRANSLATE`
    - `WIDTH_BUCKET`

- Added support for binary data with the following functions:
    - `LEFT`
    - `LEN`
    - `LENGTH`
    - `LPAD`
    - `REVERSE`
    - `RIGHT`
    - `RPAD`
    - `SUBSTR`
    - `SUBSTRING`


- Added support for the following window/aggregation functions
    - `ANY_VALUE`
    - `COUNT_IF`
    - `CONDITIONAL_CHANGE_EVENT`
    - `CONDITIONAL_TRUE_EVEN`
