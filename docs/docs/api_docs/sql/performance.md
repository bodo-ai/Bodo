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

  * The Snowflake User passed to Bodo does not have permissions to determine the view definition.
  * The Snowflake User passed to Bodo does not have permissions to
    read all of the underlying tables.
  * The view is a materialized or secure view.

If for any reason Bodo is unable to expand the view, then the query will execute treating
the view as a table and delegate it to Snowflake.
