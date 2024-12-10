# ROUND

`#!sql ROUND(X[, num_decimal_places])`

Rounds X to the specified number of decimal places.
By default, rounds to 0 decimal places.

For fixed-point decimals, usage follows that of Snowflake's `half_away_from_zero` rounding mode.
See the [Snowflake documentation](https://docs.snowflake.com/en/sql-reference/functions/round#usage-notes) for more details.
