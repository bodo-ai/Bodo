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
section is the window's "frame" used to specify the window over which to compute the
function. A bound can can come before the current row, using `#!sql PRECEDING` or after the current row, using
`#!sql FOLLOWING`. The bounds can be relative (i.e.
`#!sql N PRECEDING` or `#!sql N FOLLOWING`), where `N` is a positive integer,
or they can be absolute (i.e. `#!sql UNBOUNDED PRECEDING` or
`#!sql UNBOUNDED FOLLOWING`).

For example, consider the following window function calls:

```sql
SELECT
    SUM(A) OVER () as S1,
    SUM(A) OVER (PARTITION BY B) as S2,
    SUM(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as S3,
    SUM(A) OVER (ORDER BY C ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as S4,
FROM table1
```
This query computes 4 sums, and returns 1 row for every row in the original table:

- `S1`: The sum of each row of `A` across the entire table. 
- `S2`: The sum of all values of `A` within each partition of `B`.
- `S3`: The cumulative sum of each row and all rows before it when the rows are partitioned by B the sorted by C.
- `S4`: The the sum each row with the row before & after it in the entire table when ordered by C.

!!! note
    For most window functions, BodoSQL returns `NULL` if the specified window frame
    is empty or all `NULL`. Exceptions to this behavior are noted.

All window functions optionally allow `#!sql PARTITION BY`. Some window functions optionally allow `#!sql ORDER BY`, and some may actually require it. Some window functions optionally allow window frames, and others ban it. If a function supports window frames but one is not provided, the default frame behavior depends on the function.

!!! note
    `#!sql RANGE BETWEEN` is not currently supported.

!!!note
    If a window frame contains `NaN` values, the output may diverge from Snowflake's
    behavior. When a `NaN` value enters a window, any window function that combines
    the results with arithmetic (e.g. `SUM`, `AVG`, `VARIANCE`, etc.) will output
    `NaN` until the `NaN` value has exited the window.

BodoSQL Currently supports the following Aggregation & Window functions:

| Function | Supported with GROUP BY? | Supported without GROUP BY? | Supported as window function?  | (WINDOW) Requires ORDER BY? | (WINDOW) Allows frame? |
|---|---|---|---|---|---|
| `#!sql ANY_VALUE` | Y | Y | Y | N | N |
| `#!sql APPROX_PERCENTILE` | N | Y | Y | N | N |
| `#!sql ARRAY_AGG` | Y | N | N | N/A | N/A |
| `#!sql ARRAY_UNIQUE_AGG` | Y | N | N | N/A | N/A |
| `#!sql AVG` | Y | Y | Y | N | Y |
| `#!sql BITAND_AGG` | Y | Y | Y | N | N |
| `#!sql BITOR_AGG` | Y | Y | Y | N | N |
| `#!sql BITXOR_AGG` | Y | Y | Y | N | N |
| `#!sql BOOLAND_AGG` | Y | Y | Y | N | N |
| `#!sql BOOLOR_AGG` | Y | Y | Y | N | N |
| `#!sql BOOLXOR_AGG` | Y | Y | Y | N | N |
| `#!sql CONDITIONAL_CHANGE_EVENT` | N | N | Y | Y | N |
| `#!sql CONDITIONAL_TRUE_EVENT` | N | N | Y | Y | N |
| `#!sql CORR` | N | N | Y | N | N |
| `#!sql COUNT` | Y | Y | Y | N | Y |
| `#!sql COUNT(*)` | Y | Y | Y | N | Y |
| `#!sql COUNT_IF` | Y | Y | Y | N | Y |
| `#!sql COVAR_POP` | N | N | Y | N | N |
| `#!sql COVAR_SAMP` | N | N | Y | N | N |
| `#!sql CUME_DIST` | N | N | Y | Y | N |
| `#!sql DENSE_RANK` | N | N | Y | Y | N |
| `#!sql FIRST_VALUE` | N | N | Y | N | Y |
| `#!sql KURTOSIS` | Y | Y | Y | N | N |
| `#!sql LEAD` | N | N | Y | Y | N |
| `#!sql LAST_VALUE` | N | N | Y | N | Y |
| `#!sql LAG` | N | N | Y | Y | N |
| `#!sql LISTAGG` | Y | Y | N | N/A | N/A |
| `#!sql MAX` | Y | Y | Y | N | Y |
| `#!sql MEDIAN` | Y | Y | Y | N | N |
| `#!sql MIN` | Y | Y | Y | N | Y |
| `#!sql MODE` | Y | N | Y | N | N |
| `#!sql NTH_VALUE` | N | N | Y | N | Y |
| `#!sql NTILE` | N | N | Y | Y | N |
| `#!sql OBJECT_AGG` | Y | N | Y | N | N | N |
| `#!sql PERCENTILE_CONT` | Y | Y | N | N/A | N/A |
| `#!sql PERCENTILE_DISC` | Y | Y | N | N/A | N/A |
| `#!sql PERCENT_RANK` | N | N | Y | Y | N |
| `#!sql RANK` | N | N | Y | Y | N |
| `#!sql RATIO_TO_REPORT` | N | N | Y | N | N |
| `#!sql ROW_NUMBER` | N | N | Y | Y | N |
| `#!sql SKEW` | Y | Y | Y | N | N |
| `#!sql STDDEV` | Y | Y | Y | N | Y |
| `#!sql STDDEV_POP` | Y | Y | Y | N | Y |
| `#!sql STDDEV_SAMP` | Y | Y | Y | N | Y |
| `#!sql SUM` | Y | Y | Y | N | Y |
| `#!sql VARIANCE` | Y | Y | Y | N | Y |
| `#!sql VARIANCE_POP` | Y | Y | Y | N | Y |
| `#!sql VARIANCE_SAMP` | Y | Y | Y | N | Y |
| `#!sql VAR_POP` | Y | Y | Y | N | Y |
| `#!sql VAR_SAMP` | Y | Y | Y | N | Y |
