# Inlining

Inlining allows the compiler to perform optimizations across functions,
at the cost of increased compilation time. Use inlining when you have
your code split into multiple Bodo functions and there are important
optimizations that need to performed on some functions, that are
dependent on the code of other functions. We will explain this with
examples below.

!!! danger
    Inlining should be used sparingly as it can cause increased compilation
    time. We strongly recommend against inlining functions with 10 or more
    lines of code.


Bodo's compiler translates high-level code inside `bodo.jit` decorated
functions to highly optimized lower level code. It can perform many
optimizations on the generated code based on the structure of the code
inside the function being compiled.

Let's consider the following example where `data.pq` is a dataset with
1000 columns:

``` py
@bodo.jit
def example():
    df = pd.read_parquet("data.pq")
    return df.groupby("A")["B", "C"].sum()
```

To execute the query inside the `example` function, Bodo doesn't need
to read all the columns from the file. It only needs three columns (`A`,
`B` and `C`), and can save a lot of time and memory by just reading
those. When compiling `example`, Bodo automatically optimizes the
`read_parquet` call to only read the three required columns.

!!! warning
    If you have separate Bodo functions and their code needs to be optimized
    jointly, you need to use inlining.


Any code that needs to be optimized jointly needs to be compiled as part
of the same JIT compilation. If we have the following:

``` py
@bodo.jit
def read_data(fname):
    return pd.read_parquet(fname)

@bodo.jit
def query():
    df = read_data("data.pq")
    return df.groupby("A")["B", "C"].sum()
```

Bodo will compile the functions separately, and won't be able to
optimize the `read_parquet` call because it doesn't know how the return
value of `read_data` is used. To structure the code into different
functions and still allow the compiler to do holistic optimizations,
you can specify the `inline="always"` option to the jit decorator to
tell the compiler to include that function during compilation of another
one.

For example:

``` py
@bodo.jit(inline="always")
def read_data(fname):
    return pd.read_parquet(fname)

@bodo.jit
def query():
    df = read_data("data.pq")
    return df.groupby("A")["B", "C"].sum()
```

The option `inline="always"` in this example tells the compiler to
compile and include `read_data` when it is compiling `query`.

