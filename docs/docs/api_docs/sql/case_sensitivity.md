# Identifier Case Sensitivity

In BodoSQL all identifiers not wrapped in quotes are automatically converted to upper case.
If you are a Snowflake user who is using either the Snowflake Catalog or Table Path API, then this should not impact you and the
rules will be the same as Snowflake (i.e. identifiers are case-insensitive unless wrapped in quotes during table creation).
[See here for the Snowflake documentation.](https://docs.snowflake.com/en/sql-reference/identifiers-syntax#label-identifier-casing).

This means that the following queries are equivalent:

```sql
SELECT A FROM table1
```

```sql
SELECT a FROM TABLE1
```

When providing column or table names, identifiers will only match if the original name is in uppercase
For example, the following code will fail to compile because there is no match for TABLE1:

```py
@bodo.jit
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"table1": df1})
    return bc.sql("SELECT A FROM table1")
```

To match non-uppercase names you can use quotes to specify the name exactly as it appears in the BodoSQLContext
definition or the columns of a DataFrame. For example:

```py
@bodo.jit
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"table1": df1})
    return bc.sql("SELECT A FROM \"table1\"")
```

Similarly if you want an alias to be case sensitive then you will also need it to be wrapped in quotes:

```sql
SELECT A as "myIdentifier" FROM table1
```

If you provide DataFrames directly from Python or are using the TablePath API to load Parquet files, then please be advised
that the column names will be required to match exactly and for ease of use we highly recommend using uppercase column names.

