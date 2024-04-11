# TablePath API {#table-path-api}

The `TablePath` API is a general purpose IO interface to specify IO sources. This API is meant
as an alternative to natively loading tables in Python inside JIT functions.
The `TablePath` API stores the user-defined data location and the storage type to load a table of interest.
For example, here is some sample code that loads two DataFrames from parquet using the `TablePath` API.

```py
bc = bodosql.BodoSQLContext(
    {
        "T1": bodosql.TablePath("my_file_path1.pq", "parquet"),
        "T2": bodosql.TablePath("my_file_path2.pq", "parquet"),
    }
)

@bodo.jit
def f(bc):
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")
```

Here, the `TablePath` constructor doesn't load any data. Instead, a `BodoSQLContext` internally generates code to load the tables of interest after parsing the SQL query. Note that a `BodoSQLContext` loads all used tables from I/O *on every query*, which means that if users would like to perform multiple queries on the same data, they should consider loading the DataFrames once in a separate JIT function.

## API Reference

- `bodosql.TablePath(file_path: str, file_type: str, *, conn_str: Optional[str] = None, reorder_io: Optional[bool] = None)`
<br><br>

    Specifies how a DataFrame should be loaded from IO by a BodoSQL query. This
    can only load data when used with a `BodoSQLContext` constructor.

    ***Arguments***

    - `file_path`: Path to IO file or name of the table for SQL. This must constant at compile time if used inside JIT.

    - `file_type`: Type of file to load as a string. Supported values are ``"parquet"`` and ``"sql"``. This must constant at compile time if used inside JIT.

    - `conn_str`: Connection string used to connect to a SQL DataBase, equivalent to the conn argument to `pandas.read_sql`. This must be constant at compile time if used inside JIT and must be None if not loading from a SQL DataBase.

   - `reorder_io`: Boolean flag determining when to load IO. If `False`, all used tables are loaded before executing any of the query. If `True`, tables are loaded just before first use inside the query, which often results in decreased
    peak memory usage as each table is partially processed before loading the next table. The default value, `None`, behaves like `True`, but this may change in the future. This must be constant at compile time if used inside JIT.
