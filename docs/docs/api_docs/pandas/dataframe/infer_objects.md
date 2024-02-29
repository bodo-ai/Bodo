# `pd.DataFrame.infer_objects`


`pandas.DataFrame.infer_objects()`

### Example Usage

```py
>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [1,2,3]})
...   return df.infer_objects()
   A
0  1
1  2
2  3
```
!!! note
    Bodo does not internally use the object dtype, so types are never inferred. As a result, this API just produces a deep copy, consistent with Pandas.



