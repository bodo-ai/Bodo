# Verbose Mode {#bodoverbosemode}

When compiling functions, Bodo introduces various optimizations to improve
runtime performance. Since the success of certain optimizations can be essential,
we provide the option to run Bodo in `verbose mode` and track certain optimizations
at compile time. The information provided by `verbose mode` can help you better understand you workload's performance
as well as how to debug the workload. Additionally, using Python's `logging` module alongside Bodo's `verbose mode` can be used to track optimizations across frequently running jobs.


!!! note
      Currently all of our optimizations are tracked at compile time. This information is not stored if a function is cached.

## Example Usage

To detect important optimizations, all you need to do is set a verbose level in the global scope of a Python file using `bodo.set_verbose_level(level)`. The verbose level
is a positive integer, with greater values outputting more detailed information. The optimizations that are
expected to be the most impactful are tracked at level 1, so in most situations you can just do `bodo.
set_verbose_level(1)`. More information on the optimizations that are displayed is found in the
[`set_verbose_level` API reference](#set_verbose_level). Now when Bodo compiles a function, `rank 0` will log
important optimizations to `stderr` using Python's `logging` package.

Below is an example using the `verbose mode` to verify that Bodo is only loading the 1 column from a parquet file that is actually needed as opposed to any additional columns.


```py
bodo.set_verbose_level(1)

@bodo.jit
def load_data(filename):
    df = pd.read_parquet(filename)
    return df.id

load_data("my_file.pq")
```

```console
2024-03-24 10:44:21,023 - Bodo Default Logger - INFO - 
================================================================================
--------------------------------Filter Pushdown---------------------------------
Arrow filters pushed down:
None
None


================================================================================
2024-03-24 10:44:21,023 - Bodo Default Logger - INFO - 
================================================================================
---------------------------------Column Pruning---------------------------------
Finish column pruning on read_parquet node:

File "objmode.py", line 10:
def load_data(filename):
    df = pd.read_parquet(filename)
    ^
Columns loaded ['id']

================================================================================
```

You can also log this information to a valid `logging.Logger` instance with Bodo.

!!! info "Important"
    The logger should be a variable set in a global scope.

```py
bodo.set_verbose_level(1)
logger = logging.getLogger("myLogger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("example.log"))
bodo.set_bodo_verbose_logger(logger)

@bodo.jit
def load_data(filename):
    df = pd.read_parquet(filename)
    return df.id

load_data("my_file.pq")
```

The output will be written to `example.log` in the current working directory.

```console

## Leveraging Optimizations for Debugging

Optimzation logging can be useful for diagnosing possible performance issues. Below is an example that shows the impact of printing a section of a DataFrame to inspect the result of `read_parquet`.

```py
bodo.set_verbose_level(1)

@bodo.jit
def load_data(filename):
    df = pd.read_parquet(filename)
    print(df.head(10))
    return df.id

load_data("my_file.pq")
```

```console
2024-03-24 10:48:19,046 - Bodo Default Logger - INFO - 
================================================================================
--------------------------------Filter Pushdown---------------------------------
Arrow filters pushed down:
None
None


================================================================================
2024-03-24 10:48:19,047 - Bodo Default Logger - INFO - 
================================================================================
---------------------------------Column Pruning---------------------------------
Finish column pruning on read_parquet node:

File "objmode.py", line 11:
def load_data(filename):
    df = pd.read_parquet(filename)
    ^
Columns loaded ['id', 'Hectare', 'Date', 'Age', 'Primary Fur Color', 'Highlight Fur Color', 'Location', 'Specific Location', 'Running', 'Chasing', 'Climbing', 'Eating', 'Foraging', 'Other Activities', 'Kuks', 'Quaas', 'Moans', 'Tail flags', 'Tail twitches', 'Approaches', 'Indifferent', 'Runs from', 'Other Interactions']
```

Printing `df.head()` prints every column in the DataFrame, so Bodo must load
all of the columns. In contrast, without this print, Bodo can load just a single
column from the parquet file, so this increases both memory usage and execution time.

In some situations the reason for an optimization failure may not be as straightforward as the example above. Even when the code is more complicated,
the success/failure of optimizations can be an extremely useful first step to determine why performance is worse than expected.

## User APIs

### <a name="set_verbose_level"></a> set_verbose_level

- <code><apihead>bodo.<apiname>set_verbose_level</apiname>(level)</apihead></code>
<br><br>

    Determines if compiled JIT functions should output logging information. Level 0
    disables optimization logging and level 1 contains all of the most important operations.

    The optimizations currently displayed at each level are:

    | Verbose Level | Optimizations |
    |----------------------------------|--------------------------------------|
    | 1 | <ul><li>Column Pruning</li><li>Filter Pushdown</li><li>Dictionary Encoding</li><li>Limit Pushdown</li><li>BodoSQL generated IO time</li></ul> |
    | 2 | <ul><li>Join column pruning</li></ul> |

    ***Arguments***

    - level: A non-negative integer for the logging granularity.

    **Note**: `bodo.set_verbose_level()` should not be used inside a JIT function.

### set_bodo_verbose_logger

- <code><apihead>bodo.<apiname>set_bodo_verbose_logger</apiname>(logger)</apihead></code>
<br><br>

    Sets the logging location for Bodo verbose messages. Bodo will write to this logger
    on `rank 0` only, to prevent possible conflicts when writing to an output file. All
    messages are given with `logging.info`, so the logger should have an appropriate
    effect level.

    ***Arguments***

    - `logger`: An instance of type `logging.Logger`.

    **Note**: `bodo.set_bodo_verbose_logger()` should not be used inside a JIT function.
