# User-Defined Functions (UDFs) {#udfs}

While Pandas and other APIs can be extremely expressive, many data
science and data engineering use cases require additional functionality
beyond what is directly offered. In these situations, many programmers
create *User Defined Functions*, or UDFs, which are Python
functions designed to compute on each row or groups of rows depending on
the context.

## Using UDFs with Bodo

Bodo users can construct UDFs either by defining a separate JIT function
or by creating a function within a JIT function (either via a lambda or
closure). For example, here are two ways to construct a UDF that
advances each element of a Timestamp Series to the last day of the
current month.

``` py
import pandas as pd
import bodo

@bodo.jit
def jit_udf(x):
    return x + pd.tseries.offsets.MonthEnd(n=0, normalize=True)

@bodo.jit
def jit_example(S):
    return S.map(jit_udf)

@bodo.jit
def lambda_example(S):
    return S.map(lambda x: x + pd.tseries.offsets.MonthEnd(n=0, normalize=True))

S = pd.Series(pd.date_range(start='1/1/2021', periods=100))
pd.testing.assert_series_equal(jit_example(S), lambda_example(S))
```

UDFs can be used to compute one value per row or group (map functions)
or compute an aggregation (agg functions). Bodo provides APIs for both,
which are summarized below. Please refer to
[supported Pandas API][pandas] for more
information.

### Map Functions

-   `Series.map`
-   `Series.apply`
-   `Series.pipe`
-   `DataFrame.map`
-   `DataFrame.apply`
-   `DataFrame.pipe`
-   `GroupBy.apply`
-   `GroupBy.pipe`
-   `GroupBy.transform`

### Agg Functions

-   `GroupBy.agg`
-   `GroupBy.aggregate`

## UDF Performance

Bodo offers support for UDFs without the significant runtime penalty
generally incurred in Pandas. An example of this is shown in the
[quick started guide][example_code_in_pandas].

Bodo achieves a drastic performance advantage on UDFs because UDFs can
be optimized by similar to any other JIT code. In contrast, library
based solutions are typically limited in their ability to optimize UDFs.

## Additional Arguments

We recommend passing additional variables to UDFs explicitly, instead of
directly using variables local to the function defining the UDF. The
latter is called the \"captured\" variables case, which is often
error-prone and may result in compilation errors.

For example, consider a UDF that appends a variable suffix to each
string in a Series of strings. The proper way to write this function is
to use the `args` argument to `Series.apply()`.

```py
import pandas as pd
import bodo

@bodo.jit
def add_suffix(S, suffix):
    return S.apply(lambda x, suf: x + suf, args=(suffix,))

S = pd.Series(["abc", "edf", "32", "Vew3", "er3r2"] * 10)
suffix = "_"
add_suffix(S, suffix)
```

Alternatively, arguments can be passed by keyword.

```py
@bodo.jit
def add_suffix(S, suffix):
    return S.apply(lambda x, suf: x + suf, suf=suffix)
```

!!! note
    Not all APIs support additional arguments. Please refer to
    [supported Pandas API][pandas] for more
    information on intended API usage.

## Apply with Pandas Methods and Numpy ufuncs

In addition to UDFs, the `apply` API can also be used to call Pandas
methods and Numpy ufuncs. To execute a Pandas method, you can provide
the method name as a string.

```py
import pandas as pd
import bodo

@bodo.jit
def ex(S):
    return S.apply("nunique")

S = pd.Series(list(np.arange(100) + list(np.arange(100))))
ex(S)
```

Numpy ufuncs can either be provided with a string matching the name or
with the function itself.

```py
import numpy as np
import pandas as pd
import bodo

@bodo.jit
def ex_str(S):
    return S.apply("sin")

def ex_func(S):
    return S.apply(np.sin)

S = pd.Series(list(np.arange(100) + list(np.arange(100))))
pd.testing.assert_series_equal(ex_str(S), ex_func(S))
```

!!! note
    Numpy ufuncs are not currently supported with DataFrames.


## Type Stability Restrictions

Bodo's type stability requirements can encounter some limitations when
either using `DataFrame.apply` with different column types or when
returning a DataFrame.

### Differently Typed Columns

`DataFrame.apply` maps user provided UDFs to each row of the DataFrame.
In the situation where a DataFrame has columns of different types, the
Series passed to the UDF will contain values with different types. Bodo
internally represents these as a Heterogeneous Series. This
representation has limitations in the Series operations it supports.
Please refer to the [supported operations for heterogeneous series][heterogeneous_series] for
more information.

### Returning a DataFrame

In Pandas, `Series.apply` or `DataFrame.apply` there are multiple ways
to return a DataFrame instead of a Series. However, for type stability
reasons, Bodo can only infer a DataFrame when returning a Series whose
size can be inferred at compile time for each row.

!!! note
    If you provide an Index, then all Index values must be compile time
    constants.

Here is an example using`Series.apply` to return a DataFrame.

```py
import pandas as pd
import bodo

@bodo.jit
def series_ex(S):
    return S.apply(lambda x: pd.Series((1, x)))

S = pd.Series(list(np.arange(100) + list(np.arange(100))))
series_ex(S)
```

If using a UDF that returns a DataFrame in Pandas through another means,
this behavior will not match in Bodo and may result in a compilation
error. Please convert your solution to one of the supported methods if
possible.

### Using the Engine Keyword Argument in Pandas

Starting in Pandas 3.0, users will be able to pass a Bodo jit decorator as the `engine` argument to `DataFrame.apply`, which will automatically apply the decorator when applying the UDF. For example:

```py
import pandas as pd
import bodo

def update_score(S, answer, num_points):
    if S.guess == answer:
        return S.score + num_points
    else:
        return S.score

df = pd.DataFrame(
    {
        "guess": ["A", "B", "C", "D", "A"],
        "score": [0, 3, 4, 2, 1]
    }
)

df["updated_score"] = df.apply(update_score, axis=1, args=("A", 3), engine=bodo.jit)
```

Note that the same restrictions related to type stability discussed in the previous sections apply here as well. The example above is equivalent to:
``` py
import pandas as pd
import bodo

def update_score(S, answer, num_points):
    if S.guess == answer:
        return S.score + num_points
    else:
        return S.score

@bodo.jit
def apply_update_scores(df, answer, num_points):
    return df.apply(update_score, axis=1, args=(answer, num_points))

df = pd.DataFrame(
    {
        "guess": ["A", "B", "C", "D", "A"],
        "score": [0, 3, 4, 2, 1]
    }
)

df["updated_score"] = apply_update_scores(df, "A", 3)
```
