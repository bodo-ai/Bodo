# Performance Considerations

This section discusses some factors which affect performance when using BodoSQL.

## Snowflake Views

Users may define views within their Snowflake account to enable greater query reuse.
Views may constitute performance bottlenecks because if a view is evaluated in Snowflake
Bodo will need to wait for the result before it can fetch data and may have less access
to optimizations.

To improve performance in these circumstances Bodo will attempt to expand any views into
the body of the query to allow Bodo to operate on the underlying tables. When this occurs
users should face no performance penalty for using views in their queries. However there are
a few situations in which this is not possible, namely

- The Snowflake User passed to Bodo does not have permissions to determine the view definition.
- The Snowflake User passed to Bodo does not have permissions to
  read all of the underlying tables.
- The view is a materialized or secure view.

If for any reason Bodo is unable to expand the view, then the query will execute treating
the view as a table and delegate it to Snowflake.

## Vectorized Execution

Bodo uses a vectorized query execution model for SQL queries. In this model, the query plan is split into several stages or pipelines, each comprised of multiple operators. Each operator of the pipeline performs computations on batches of data and passes on the result to the next operator. This enables several performance and reliability benefits:

- It is possible to overlap compute and I/O. When reading data, BodoSQL can start executing compute tasks on the chunks of data that have been read while the rest is waiting on I/O. Similarly, BodoSQL can start writing the chunks of data that have finished being computed while the rest is still in progress.
- There is increased CPU cache locality when working on smaller chunks, potentially increasing runtime performance.
- This prevents out-of-memory (OOM) errors for most use cases. For example, in simple aggregations, data can be read and aggregated incrementally, instead of needing to read all the data and then performing the aggregation on the entire table.

For operators like Join, Aggregate, Sort and some Window functions, we use specialized operators that can spill to disk for reliable execution even when the intermediates are too big to fit in memory.

Note that not every operator in BodoSQL fully supports vectorized execution yet. Some operations still require BodoSQL to have the entire data in main memory before starting any compute.

The best way to determine if a query is using vectorized execution in BodoSQL is to check the query plan
with `bc.generate_plan(query)`. BodoSQL generates two special RelNodes when it needs to break vectorized
execution: `CombineStreamsExchange` is used to gather the chunked batches of data into a single in-memory table
(potentially causing an OOM error if the table is big enough) so that whatever operators occur next can operate
on the non-streamed data, and `SeparateStreamsExchange` is used to split up such a table back into multiple
chunks so that subsequent operators can use vectorized execution.

For example, consider the abstract BodoSQL plan below:

```
TableCreate
  NodeA
    SeparateStreamsExchange
      NodeB
        CombineStreamsExchange
          TableScan
```

At runtime, BodoSQL will perform the table scan in batches, and as it receives each batch it will
combine them into a single table. Once that singular table has been created, the operation described
by `NodeB` is run on the entire table. Afterward, the output of `NodeB` is split up into smaller chunks
that are fed into `NodeA`, which processes each chunk one at a time. Finally, the output of `NodeA`
is fed into the table create operation, which writes each chunk as it receives it rather than waiting
for `NodeA` to finish processing every chunk.

Some notable examples of operations that do not currently use vectorized execution:

- Some set operators (`INTERSECT` and `MINUS`).
- Aggregations without `GROUP BY`.

Additionally, BodoSQL's support for vectorized execution of window functions is a limited and experimental feature.
