# Compilation Tips and Troubleshooting {#compilation}

## What Code to JIT Compile

The general recommendation is to use Bodo JIT compilation only for code
that is data and/or compute intensive (e.g. Pandas code on large
dataframes). In other words:

-   Only use Bodo for data processing and analytics code such as Pandas, Numpy, and Scikit-Learn 
    (see [Bodo API reference][apireference] for analytics APIs with JIT support).
-   Refactor code that sets up infrastructure or performs initializations out of JIT functions.

This reduces the risk of encountering unsupported features and also
reduces compilation time. For example, the program below finds the input
file name in regular Python, and uses Bodo JIT *only* for data load and
processing:

```py
import bodo
import pandas as pd
import os

def get_filename():
    if os.path.exists("input.parquet"):
        return "input.parquet"
    if "INPUT_FILE" in os.environ:
        return os.environ["INPUT_FILE"]
    raise Exception("Input file name not found")

@bodo.jit
def f(fname):
    df = pd.read_parquet(fname)
    print(df.sum())

fname = get_filename()
f(fname)
```

This recommendation is similar to Numba's
[What to compile](https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html#what-to-compile).

## Compilation Errors {#whycompilationerror}

First of all, let us understand why the code may fail to compile. There
are three main kinds of issues:

1.  Some API is used that is not supported in Bodo JIT yet (see [Bodo API Reference][apireference]).
2.  Some Python construct or data structure is used that cannot be JIT compiled
    (see [Unsupported Python APIs][notsupportedpython]).
3.  The code has type stability issues (see [type stability][typestability]).

Below are some examples of the type of errors you may see due to these
issues.

### Unsupported Functions or Methods

If a JIT function uses an unsupported function or method (e.g. in Pandas
APIs), Bodo raises `BodoError` explaining that the method is not
supported yet:

```
BodoError: <method> not supported yet
```

For example:

```py
>>> @bodo.jit
... def f(df):
...     return df.swapaxes(0, 1)
...
>>> f(df)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 874, in _compile_for_args
    raise error
bodo.utils.typing.BodoError: DataFrame.swapaxes() not supported yet

```

### Unsupported Attributes

Attempting to access an unsupported attribute in Bodo JIT functions will
result in a `BodoError` as follows:

```
BodoError: <attribute> not supported yet
```
        
For example:

```py
>>> @bodo.jit
... def f(df):
...     return df.flags
...
>>> f(df)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 874, in _compile_for_args
    raise error
bodo.utils.typing.BodoError: DataFrame.flags not supported yet

```

### Unsupported Arguments

Supported APIs may not support all optional arguments. Supplying an
unsupported argument will result in a `BodoError`:

```
BodoError: <method>: <keyword> argument not supported yet
```

For example:

```py
>>> @bodo.jit
... def f(df):
...     return df.sort_index(key=lambda x: x.str.lower())
...
>>> f(df)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 874, in _compile_for_args
    raise error
bodo.utils.typing.BodoError: DataFrame.sort_index(): key parameter only supports default value None
Please check supported Pandas operations here (https://docs.bodo.ai/latest/api_docs/pandas/dataframe/).

```

## Type Stability Errors

Bodo needs to infer data types for all program variables for successful
JIT compilation. A type stability issue arises when different program
control flow paths assign values with different types to a variable. For
example, variable `a` below could either be an integer or a string:

```py
>>> @bodo.jit
... def f(flag):
...     if flag:
...         a = 3
...     else:
...         a = "A"
...     return a
...
>>> f(True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 874, in _compile_for_args
    raise error
bodo.utils.typing.BodoError: Unable to unify the following function return types: [Literal[int](3), Literal[str](A)]
```

The error `TypingError: Cannot unify <type1> and <type2>` means that the
two possible data types cannot be combined and therefore, the variable
cannot have a single data type.

Dataframe variables require their schema (column names and their types)
to be consistent for type stability (see
[dataframe schema stability][schemastability]). For example, the
dataframe variable `df` below could either have a single column ("A":
integer) or two columns ("A": integer, "B": float) depending on the
runtime value of `flag`, which results in a type stability error:

```py
>>> @bodo.jit
... def f(flag):
...     df = pd.DataFrame({"A": [1, 2, 3, 4]})
...     if flag:
...         df["B"] = [1.2, 0.4, 0.7, 121.9]
...     print(df)
...
>>> f(True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 854, in _compile_for_args
    error_rewrite(e, 'typing')
  File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 763, in error_rewrite
    raise e.with_traceback(None)
numba.core.errors.TypingError: Cannot unify dataframe((Array(int64, 1, 'C', False, aligned=True),), RangeIndexType(none), ('A',), 1D_Block_Var, False, False) and dataframe((Array(int64, 1, 'C', False, aligned=True), Array(float64, 1, 'C', False, aligned=True)), RangeIndexType(none), ('A', 'B'), 1D_Block_Var, False, False) for 'df', defined at <stdin> (3)


```

Additionally, some function arguments need to be constant to ensure type
stability. In certain cases where it is possible, Bodo may infer the
constant values. In other cases, it may throw an error indicating that
the argument should be constant. For instance, `axis` argument in
`pd.concat` determines whether the output is a Series type or a
dataframe type in the example below. Therefore, Bodo needs to know the
value at compilation time for type inference. Otherwise, an error is
thrown (passing `axis` as argument to the JIT function fixes the error
in this case):

```py
>>> import pandas as pd
>>> import bodo
>>> @bodo.jit
... def f(S1, S2, flag):
...     axis = 0
...     if flag:
...         axis = 1
...     return pd.concat([S1, S2], axis=axis)
...
>>> S1 = pd.Series([1, 2, 3], name="A")
>>> S2 = pd.Series([3, 4, 5], name="B")
>>> f(S1, S2, False)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 874, in _compile_for_args
    raise error
bodo.utils.typing.BodoError: pd.concat(): 'axis' should be a constant integer


>>> @bodo.jit
... def f(S1, S2, axis):
...     return pd.concat([S1, S2], axis=axis)
...
>>> print(f(S1, S2, 0))

    0    1
    1    2
    2    3
    0    3
    1    4
    2    5
    dtype: int64
```
See [Bodo API reference][apireference] for more details on argument requirements.

## Troubleshooting Compilation Errors

Now that we understand what causes the error, let's fix it!

For potential unsupported APIs, Python feature gaps or type stability
issues try the following:

1.  Make sure your code works in Python. In a lot of cases, a Bodo
    decorated function does not compile, but it does not compile in
    Python either.

2.  Refactor your code with supported operations if possible. For
    instance, the `sort_index(key=lambda ...)` examble above can be
    replaced with regular `sort_values`:

    ```py
    >>> df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=['A', 'b', 'C', 'd'])
    >>> @bodo.jit
    ... def f(df):
    ...     return df.sort_index(key=lambda x: x.str.lower())
    ...
    >>> f(df)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 874, in _compile_for_args
        raise error
    bodo.utils.typing.BodoError: DataFrame.sort_index(): key parameter only supports default value None

    >>> @bodo.jit
    ... def f(df):
    ...     df["key"] = df.index.map(lambda a: a.lower())
    ...     return df.sort_values("key").drop(columns="key")
    ...
    >>> f(df)
        a
        A  1
        b  2
        C  3
        d  4
    ```
    
3.  Refactor your code and use regular Python for unsupported
    features.

    a.  Move the code causing issues to regular Python and pass
        necessary data to JIT functions.

    b.  Use `@bodo.wrap_python` to perform some computation within JIT
        functions in regular Python if necessary (see [@bodo.wrap_python][objmode]).

4.  Refactor your code to make it type stable (see
    [type stability][typestability]). For example:

    ```py
    >>> flag = True
    >>> @bodo.jit
    ... def f(flag):
    ...     df = pd.read_parquet("in.parquet")
    ...     if flag:
    ...             df["C"] = 1
    ...     df.to_parquet("out.parquet")
    ...
    >>> f(flag)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 854, in _compile_for_args
        error_rewrite(e, 'typing')
      File "/opt/miniconda3/envs/Bodo/lib/python3.12/site-packages/bodo/numba_compat.py", line 763, in error_rewrite
        raise e.with_traceback(None)
    numba.core.errors.TypingError: Cannot unify dataframe((Array(datetime64[ns], 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True)), RangeIndexType(none), ('A', 'B'), 1D_Block_Var, True, False) and dataframe((Array(datetime64[ns], 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True), Array(int64, 1, 'C', False, aligned=True)), RangeIndexType(none), ('A', 'B', 'C'), 1D_Block_Var, True, False) for 'df', defined at <stdin> (3)



    >>> @bodo.jit
    ... def f1():
    ...     df = pd.read_parquet("in.parquet")
    ...     return df
    ...
    >>> @bodo.jit
    ... def f2(df):
    ...     df["C"] = 1
    ...     return df
    ...
    >>> @bodo.jit
    ... def f3(df):
    ...     df.to_parquet("out.parquet")
    ...
    >>> df = f1()
    >>> if flag:
    ...     df = f2(df)
    ...
    >>> f3(df)
    ```

## Disabling Python Output Buffering

Sometimes standard output prints may not appear when the program fails,
due to Python's I/O buffering. Therefore, setting `PYTHONUNBUFFERED`
environment variable is recommended for debugging:

```py
export PYTHONUNBUFFERED=1
```

## Requesting Unsupported Functionality and Reporting Errors

If you want to request a new feature, or report a bug you have found,
please create an issue in our [GitHub repository](https://github.com/bodo-ai/Bodo). If you
encounter an error which is not covered on this page, please report it
to our repository as well.

