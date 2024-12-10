# BodoSQLContext API

The `BodoSQLContext` API is the primary interface for executing SQL queries. It performs two roles:

1. Registering data and connection information to load tables of interest.
1. Forwarding SQL queries to the BodoSQL engine for compilation and execution. This is done via the
   `bc.sql(query)` method, where `bc` is a `BodoSQLContext` object.

A `BodoSQLContext` can be defined in regular Python and passed as an argument to JIT functions or can be
defined directly inside JIT functions. We recommend defining and modifying a `BodoSQLContext` in regular
Python whenever possible.

For example:

```py
bc = bodosql.BodoSQLContext(
    {
        "T1": bodosql.TablePath("my_file_path.pq", "parquet"),
    },
    catalog=bodosql.SnowflakeCatalog(
        username,
        password,
        account_name,
        warehouse_name,
        database name,
    )
)

@bodo.jit
def f(bc):
    return bc.sql("select t1.A, t2.B from t1, catalogSchema.t2 where t1.C > 5 and t1.D = catalogSchema.t2.D")
```

## API Reference

- `bodosql.BodoSQLContext(tables: Optional[Dict[str, Union[pandas.DataFrame|TablePath]]] = None, catalog: Optional[DatabaseCatalog] = None)`
  <br><br>

  Defines a `BodoSQLContext` with the given local tables and catalog.

  ***Arguments***

  - `tables`: A dictionary that maps a name used in a SQL query to a `DataFrame` or `TablePath` object.

  - `catalog`: A `DatabaseCatalog` used to load tables from a remote database (e.g. Snowflake).

- `bodosql.BodoSQLContext.sql(self, query: str, params_dict: Optional[Dict[str, Any] = None, distributed: list|set|bool = set(), replicated: list|set|bool = set(), **jit_options)`
  <br><br>

  Executes a SQL query using the tables registered in this `BodoSQLContext`.

  ***Arguments***

  - `query`: The SQL query to execute. This function generates code that is compiled so the `query` argument is required
    to be a compile time constant.

  - `params_dict`: A dictionary that maps a SQL usable name to Python variables. For more information please
    refer to [the BodoSQL named parameters section][bodosql_named_params].

  - `distributed`, `replicated`, and other JIT options are passed to Bodo JIT. See [Bodo distributed flags documentation](#dist-flags) for more details.
    Example code:

  ```py
  df = pd.DataFrame({"A": np.arange(10), "B": np.ones(10)})
  bc = bodosql.BodoSQLContext({"T1": df})
  out_df = bc.sql("select sum(B) from T1 group by A", distributed=["T1"])
  ```

  ***Returns***

  A `DataFrame` that results from executing the query.

- `bodosql.BodoSQLContext.add_or_replace_view(self, name: str, table: Union[pandas.DataFrame, TablePath])`
  <br><br>

  Create a new `BodoSQLContext` from an existing `BodoSQLContext` by adding or replacing a table.

  ***Arguments***

  - `name`: The name of the table to add. If the name already exists references to that table
    are removed from the new context.

  - `table`: The table object to add. `table` must be a `DataFrame` or `TablePath` object.

  ***Returns***

  A new `BodoSQLContext` that retains the tables and catalogs from the old `BodoSQLContext` and inserts the new table specified.

  !!! note
  This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
  e.g. `bc = bc.add_or_replace_view("t1", table)`

- `bodosql.BodoSQLContext.remove_view(self, name: str)`
  <br><br>

  Creates a new `BodoSQLContext` from an existing context by removing the table with the
  given name. If the name does not exist, a `BodoError` is thrown.

  ***Arguments***

  - `name`: The name of the table to remove.

  ***Returns***

  A new `BodoSQLContext` that retains the tables and catalogs from the old `BodoSQLContext` minus the table specified.

  !!! note
  This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
  e.g. `bc = bc.remove_view("t1")`

- `bodosql.BodoSQLContext.add_or_replace_catalog(self, catalog: DatabaseCatalog)`
  <br><br>

  Create a new `BodoSQLContext` from an existing context by replacing the `BodoSQLContext` object's `DatabaseCatalog` with
  a new catalog.

  ***Arguments***

  - `catalog`: The catalog to insert.

  ***Returns***

  A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but replaces the old catalog with the new catalog specified.

  !!! note
  This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
  e.g. `bc = bc.add_or_replace_catalog(catalog)`

- `bodosql.BodoSQLContext.remove_catalog(self)`
  <br><br>

  Create a new `BodoSQLContext` from an existing context by removing its `DatabaseCatalog`.

  ***Returns***

  A new `BodoSQLContext` that retains tables from the old `BodoSQLContext` but removes the old catalog.

  !!!note
  This **DOES NOT** update the given context. Users should always use the `BodoSQLContext` object returned from the function call.
  e.g. `bc = bc.remove_catalog()`
