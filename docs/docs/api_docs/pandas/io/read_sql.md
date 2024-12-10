# pd.read_sql

`pandas.read_sql`

- [example usage and more system specific instructions][sql-section]
- Argument `sql` is supported but only as a string form.
  SQLalchemy `Selectable` is not supported. There is
  no restriction on the form of the sql request.
- Argument `con` is supported but only as a string form.
  SQLalchemy `connectable` is not supported.
- Argument `index_col` is supported.
- Arguments `chunksize`, `column`, `coerce_float`, `params` are
  not supported.
