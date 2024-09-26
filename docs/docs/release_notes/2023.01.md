Bodo 2023.1 Release (Date: 01/06/2023) {#January_2023}
========================================


## New Features and Improvements

Bodo:

- Added support for returning timezone aware timestamp scalars with DataFrame attributes iat and iloc.
- Support for comparison operators between timezone aware Timestamp values in Series, array, and scalars.
- Improved the performance of coalesce for string columns.
- Improved performance on dictionary encoded columns for coalesce and support for dictionary encoded output.
- Support for tz-aware data in pd.concat.
- pd.merge now supports cross join type (how=”cross”). Cross joins in BodoSQL are now significantly faster and more scalable.
- Supported passing timezone to pd.Timestamp.now().
- Support for comparison operators between Timestamps (both timezone aware and timezone naive) and datetime.date values in Series, array, and scalars.


BodoSQL:

Added support for the following functions:

- CURRENT_TIMESTAMP
- DATE_PART
- GETDATE
- DATEADD (in the form DATEADD(unit_string_literal, integer_amount, starting_datetime))
- TIMEADD
- TO_BOOLEAN / TRY_TO_BOOLEAN
- TO_CHAR / TO_VARCHAR
- CHARINDEX (not supported for binary data)
- POSITION (only in the form POSITION(X IN Y), not supported for binary data)
- INSERT (behavior when numerical arguments are negative is currently not well defined)
- STARTSWITH / ENDSWITH
- RTRIMMED_LENGTH
- MODE (only as a window function, not as a generic aggregation)
- RATIO_TO_REPORT
- BOOLOR_AGG

Parity Improvements:

- Added support for timezone aware data in all BodoSQL datetime functions.
- Adjusted TIMESTAMPDIFF to obey Snowflake SQL rounding rules (i.e. ignoring all units smaller than the selected unit).
- Support for loading views and other non-standard tables with the SnowflakeCatalog.
- Support for all join types offered by Snowflake.
- Support for tz-aware data outputs in Case statements.

Other Improvements:

- Multiple top-level calls to window functions will now compile faster in BodoSQL if they use the same partition and order.
- Snowflake writes with df.to_sql can now use the more performant direct upload strategy for Azure based Snowflake accounts.
- Snowflake I/O (read and write) no longer requires the snowflake-sqlalchemy package.
- Improved performance for reading string data in compressed format from Snowflake.
- Performance Warning if running Bodo and Snowflake in different cloud regions.
- Added support for returning timezone aware timestamp scalars with Series attributes iat, iloc, loc and regular getitem.

