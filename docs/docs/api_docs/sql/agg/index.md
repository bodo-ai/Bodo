---
hide:
  - toc
---
# Aggregation & Window Functions

An aggregation function can be used to combine data across many
rows to form a single answer. Aggregations can be done with a
`#!sql GROUP BY` clause, in which case the combined value is
calculated once per unique combination of groupbing keys. Aggregations
can also be done without the `#!sql GROUP BY` clause, in which case
a single value is outputted by calculating the aggregation across
all rows.

For example:

```sql
SELECT AVG(A) FROM table1 GROUP BY B

SELECT COUNT(Distinct A) FROM table1
```

Window functions can be used to compute an aggregation across a
row and its surrounding rows. Most window functions have the
following syntax:

```sql
SELECT WINDOW_FN(ARG1, ..., ARGN) OVER (PARTITION BY PARTITION_COLUMN_1, ..., PARTITION_COLUMN_N ORDER BY SORT_COLUMN_1, ..., SORT_COLUMN_N ROWS BETWEEN <LOWER_BOUND> AND <UPPER_BOUND>) FROM table_name
```
The `#!sql ROWS BETWEEN ROWS BETWEEN <LOWER_BOUND> AND <UPPER_BOUND>`
section is used to specify the window over which to compute the
function. A bound can can come before the current row, using `#!sql PRECEDING` or after the current row, using
`#!sql FOLLOWING`. The bounds can be relative (i.e.
`#!sql N PRECEDING` or `#!sql N FOLLOWING`), where `N` is a positive integer,
or they can be absolute (i.e. `#!sql UNBOUNDED PRECEDING` or
`#!sql UNBOUNDED FOLLOWING`).

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

!!! note
    For most window functions, BodoSQL returns `NULL` if the specified window frame
    is empty or all `NULL`. Exceptions to this behavior are noted.

Window functions perform a series of steps as followed:

1.  Partition the data by `#!sql PARTITION_COLUMN`. This is effectively a groupby operation on `#!sql PARTITION_COLUMN`.
2.  Sort each group as specified by the `#!sql ORDER BY` clause.
3.  Perform the calculation over the specified window, i.e. the newly ordered subset of data.
4.  Shuffle the data back to the original ordering.

For BodoSQL, `#!sql PARTITION BY` is required, but
`#!sql ORDER BY` is optional for most functions and
`#!sql ROWS BETWEEN` is optional for all of them. If
`#!sql ROWS BETWEEN` is not specified then it defaults to either
computing the result over the entire window (if no `#!sql ORDER BY`
clause is specified) or to using the window `#!sql UNBOUNDED PRECEDING TO CURRENT ROW`
(if there is an `#!sql ORDER BY` clause).
!!! note
    `#!sql RANGE BETWEEN` is not currently supported.

!!!note
    If a window frame contains `NaN` values, the output may diverge from Snowflake's
    behavior. When a `NaN` value enters a window, any window function that combines
    the results with arithmetic (e.g. `SUM`, `AVG`, `VARIANCE`, etc.) will output
    `NaN` until the `NaN` value has exited the window.

BodoSQL Currently supports the following Aggregation & Window functions:

| Function | Supported with GROUP BY? | Supported without GROUP BY? | Supported as window function? | (WINDOW) Allows ORDER BY? | (WINDOW) Requires ORDER BY? | (WINDOW) Allows frame? |
|---|---|---|---|---|---|---|
| `#!sql ANY_VALUE` | Y | Y | Y | Y | N | Y |
| `#!sql APPROX_PERCENTILE` | N | Y | Y | N | N | N |
| `#!sql ARRAY_AGG` | Y | N | N | N/A | N/A | N/A |
| `#!sql ARRAY_UNIQUE_AGG` | Y | N | N | N/A | N/A | N/A |
| `#!sql AVG` | Y | Y | Y | Y | N | Y |
| `#!sql BITAND_AGG` | Y | Y | Y | N | N | N |
| `#!sql BITOR_AGG` | Y | Y | Y | N | N | N |
| `#!sql BITXOR_AGG` | Y | Y | Y | N | N | N |
| `#!sql BOOLAND_AGG` | Y | Y | Y | N | N | N |
| `#!sql BOOLOR_AGG` | Y | Y | Y | N | N | N |
| `#!sql BOOLXOR_AGG` | Y | Y | Y | N | N | N |
| `#!sql CONDITIONAL_CHANGE_EVENT` | N | N | Y | Y | Y | N |
| `#!sql CONDITIONAL_TRUE_EVENT` | N | N | Y | Y | Y | N |
| `#!sql CORR` | N | N | Y | N | N | N |
| `#!sql COUNT` | Y | Y | Y | Y | N | Y |
| `#!sql COUNT(*)` | Y | Y | Y | Y | N | Y |
| `#!sql COUNT_IF` | Y | Y | Y | Y | N | Y |
| `#!sql COVAR_POP` | N | N | Y | N | N | N |
| `#!sql COVAR_SAMP` | N | N | Y | N | N | N |
| `#!sql CUME_DIST` | N | N | Y | Y | Y | N |
| `#!sql DENSE_RANK` | N | N | Y | Y | Y | N |
| `#!sql FIRST_VALUE` | N | N | Y | Y | N | Y |
| `#!sql KURTOSIS` | Y | Y | Y | N | N | N |
| `#!sql LEAD` | N | N | Y | Y | Y | N |
| `#!sql LAST_VALUE` | N | N | Y | Y | N | Y |
| `#!sql LAG` | N | N | Y | Y | Y | N |
| `#!sql LISTAGG` | Y | Y | N | N/A | N/A | N/A |
| `#!sql MAX` | Y | Y | Y | Y | N | Y |
| `#!sql MEDIAN` | Y | Y | Y | N | N | N |
| `#!sql MIN` | Y | Y | Y | Y | N | Y |
| `#!sql MODE` | Y | N | Y | Y | N | N |
| `#!sql NTH_VALUE` | N | N | Y | Y | N | Y |
| `#!sql NTILE` | N | N | Y | Y | Y | N |
| `#!sql OBJECT_AGG` | Y | N | Y | N | N | N |
| `#!sql PERCENTILE_CONT` | Y | Y | N | N/A | N/A | N/A |
| `#!sql PERCENTILE_DISC` | Y | Y | N | N/A | N/A | N/A |
| `#!sql PERCENT_RANK` | N | N | Y | Y | Y | N |
| `#!sql RANK` | N | N | Y | Y | Y | N |
| `#!sql RATIO_TO_REPORT` | N | N | Y | Y | N | N |
| `#!sql ROW_NUMBER` | N | N | Y | Y | Y | N |
| `#!sql SKEW` | Y | Y | Y | Y | N | N |
| `#!sql STDDEV` | Y | Y | Y | Y | N | Y |
| `#!sql STDDEV_POP` | Y | Y | Y | Y | N | Y |
| `#!sql STDDEV_SAMP` | Y | Y | Y | Y | N | Y |
| `#!sql SUM` | Y | Y | Y | Y | N | Y |
| `#!sql VARIANCE` | Y | Y | Y | Y | N | Y |
| `#!sql VARIANCE_POP` | Y | Y | Y | Y | N | Y |
| `#!sql VARIANCE_SAMP` | Y | Y | Y | Y | N | Y |
| `#!sql VAR_POP` | Y | Y | Y | Y | N | Y |
| `#!sql VAR_SAMP` | Y | Y | Y | Y | N | Y |
