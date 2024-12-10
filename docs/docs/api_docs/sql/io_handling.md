# IO Handling

BodoSQL is great for compute based SQL queries, but you cannot yet access external storage directly from SQL. Instead, you can load and store data using Bodo and various Python APIs. Here we explain a couple common methods for loading data.

## Pandas IO in JIT function with SQL Query

The most common way to load data is to first use Pandas APIs to load a DataFrame inside a JIT function and then to use that DataFrame inside a BodoSQLContext.

```py
def f(f1, f2):
    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)
    bc = bodosql.BodoSQLContext(
        {
            "T1": df1,
            "T2": df2,
        }
    )
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")
```

## Pandas IO in a JIT Function Separate from Query

The previous approach works well for most individual queries. However, when running several queries on the same dataset, it should ideally be loaded once for all queries. To do this, you can structure your JIT code to contain a single load function at the beginning. For example:

```py

@bodo.jit
def load_data(f1, f2):
    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)
    return df1, df2

def q1(df1, df2):
    bc = bodosql.BodoSQLContext(
        {
            "T1": df1,
            "T2": df2,
        }
    )
    return bc.sql("select t1.A, t2.B from t1, t2 where t1.C > 5 and t1.D = t2.D")

...

@bodo.jit
def run_queries(f1, f2):
    df1, df2 = load_data(f1, f2)
    print(q1(df1, df2))
    print(q2(df2))
    print(q3(df1))
    ...

run_queries(f1, f2)
```

This approach prevents certain optimizations, such as filter pushdown. However, the assumption here is that you will use the entire DataFrame across the various benchmarks, so no optimization is useful by itself. In addition, any optimizations that can apply to all queries can be done explicitly inside `load_data`. For example, if all queries are operate on a single day's data with `df1`, you can write that filter in `load_data` to limit IO and filter pushdown will be performed.

```py

@bodo.jit
def load_data(f1, f2, target_date):
    df1 = pd.read_parquet(f1)
    # Applying this filter limits how much data is loaded.
    df1 = df1[df1.date_val == target_date]
    df2 = pd.read_parquet(f2)
    return df1, df2

@bodo.jit
def run_queries(f1, f2, target_date):
    df1, df2 = load_data(f1, f2, target_date)
    ...

run_queries(f1, f2, target_date)
```
