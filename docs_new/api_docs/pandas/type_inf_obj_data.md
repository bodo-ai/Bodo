# Type Inference for Object Data

Pandas stores some data types (e.g. strings) as object arrays which are
untyped. Therefore, Bodo needs to infer the actual data type of object
arrays when dataframes or series values are passed to JIT functions from
regular Python. Bodo uses the first non-null value of the array to
determine the type, and throws a warning if the array is empty or all
nulls:

```console
BodoWarning: Empty object array passed to Bodo, which causes ambiguity in typing. This can cause errors in parallel execution.
```

In this case, Bodo assumes the array is a string array which is the most
common. However, this can cause errors if a distributed dataset is
passed to Bodo, and some other processor has non-string data. This
corner case can usually be avoided by load balancing the data across
processors to avoid empty arrays.
