# Compile Time Constants {#require_constants}

Unlike regular Python, which is dynamically typed, Bodo needs to be able
to type all functions at compile time. While in most cases, the output
types depend solely on the input types, some APIs require knowing exact
values in order to produce accurate types.

As an example, consider the `iloc` DataFrame API. This API can be used
to selected a subset of rows and columns by passing integers or slices
of integers. A Bodo JIT version of a function calling this API might
look like:

```py
import numpy as np
import pandas as pd
import bodo

@bodo.jit
def df_iloc(df, rows, columns):
   return df.iloc[rows, columns]

df = pd.DataFrame({'A': np.arange(100), 'B': ["A", "B", "C", "D"]* 25})
print(df_iloc(df, slice(1, 4), 0))
```

If we try to run this file, we will get an error message:

```console
$ python iloc_example.py
Traceback (most recent call last):
File "iloc_example.py", line 10, in <module>
   df_iloc(df, slice(1, 4), 0)
File "/my_path/bodo/numba_compat.py", line 1195, in _compile_for_args
   raise error
bodo.utils.typing.BodoError: idx2 in df.iloc[idx1, idx2] should be a constant integer or constant list of integers

File "iloc_example.py", line 7:
def df_iloc(df, rows, columns):
   return df.iloc[rows, columns]
```

The relevant part of the error message is
`idx2 in df.iloc[idx1, idx2] should be a constant integer or constant list of integers`.

This error is thrown because depending on the value of `columns`, Bodo
selects different columns with different types. When `columns=0` Bodo
will need to compile code for numeric values, but when `columns=1` Bodo
needs to compile code for strings, so it cannot properly type this
function.

To resolve this issue, you will need to replace `columns` with a literal
integer. If instead the Bodo function is written as:

```py
import numpy as np
import pandas as pd
import bodo

@bodo.jit
def df_iloc(df, rows):
   return df.iloc[rows, 0]

df = pd.DataFrame({'A': np.arange(100), 'B': ["A", "B", "C", "D"]* 25})
print(df_iloc(df, slice(1, 4)))
```

Bodo now can see that the output DataFrame should have a single `int64`
column and it is able to compile the code.

Whenever a value needs to be known for typing purposes, Bodo will throw
an error that indicates some argument requires `a constant value`. All
of these can be resolved by making this value a literal. Alternatively,
some APIs support other ways of specifying the output types, which will
be indicated in the error message.
