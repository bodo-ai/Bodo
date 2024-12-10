# BodoSQL Caching & Parameterized Queries {#bodosql_named_params}

BodoSQL can reuse Bodo caching to avoid recompilation when used inside a
JIT function. BodoSQL caching works the same as Bodo, so for example:

```py
@bodo.jit(cache=True)
def f(filename):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"TABLE1": df1})
    df2 = bc.sql("SELECT A FROM table1 WHERE B > 4")
    print(df2.A.sum())
```

This will avoid recompilation so long as the DataFrame scheme stored in
`filename` has the same schema and the code does not change.

To enable caching for queries with scalar parameters that you may want
to adjust between runs, we introduce a feature called parameterized
queries. In a parameterized query, the SQL query replaces a
constant/scalar value with a variable, which we call a named parameter.
In addition, the query is passed a dictionary of parameters which maps
each name to a corresponding Python variable.

For example, if in the above SQL query we wanted to replace 4 with other
integers, we could rewrite our query as:

```py
bc.sql("SELECT A FROM table1 WHERE B @var", {"var": python_var})
```

Now anywhere that `@var` is used, the value of python_var at runtime
will be used instead. This can be used in caching, because python_var
can be provided as an argument to the JIT function itself, thus enabling
changing the filter without recompiling. The full example looks like
this:

```py
@bodo.jit(cache=True)
def f(filename, python_var):
    df1 = pd.read_parquet(filename)
    bc = bodosql.BodoSQLContext({"TABLE1": df1})
    df2 = bc.sql("SELECT A FROM table1 WHERE B @var", {"var": python_var})
    print(df2.A.sum())
```

Named parameters cannot be used in places that require a constant value
to generate the correct implementation (e.g. TimeUnit in EXTRACT).

!!! note
Named parameters are case sensitive, so `@var` and `@VAR` are
different identifiers.
