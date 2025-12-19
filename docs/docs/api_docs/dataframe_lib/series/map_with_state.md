# bodo.pandas.BodoSeries.map\_with\_state
```
BodoSeries.map_with_state(init_state_fn, row_fn, na_action=None, output_type=None) -> BodoSeries

```
Map values of a BodoSeries according to a mapping with a one-time initialization routine whose
result is passed to the row mapping function.

Sometimes, there is initialization code that is so expensive to run that one would like to minimize
overheads by running it just once per worker.  Other APIs such as map or map\_partitions are
not suitable for this purpose because those would require per row initialization or per partition
initialization.

!!! note
    Calling `BodoSeries.map_with_state` will immediately execute a plan and will perform an
    initialization of the state followed by running row\_fn on a small number of rows in order to
    determine the output type of the series.  This plan execution and initialization can be avoided
    if the output\_type is manually specified.

!!! note
    If a per-worker clean-up is required then state can be an instance of a class with the
    \_\_del\_\_ method defined in which the clean-up is performed.

<p class="api-header">Parameters</p>

: __init_state_fn : *function*:__ Initialization function.  Run only once per worker.

: __row_fn : *function*:__ Mapping correspondence.  The first argument to row\_fn when called is the previously initialized state variable.

: __na_actions : *{None, ‘ignore’}, default None*:__ If 'ignore' then NaN values will be propagated without passing them to the mapping correspondence.

: __output_type: *{None, Pandas.series}, default None*:__ If none, then plan is executed and sample of rows passed to row\_fn after calling init\_state\_fn to determine the output type.  This parameter can be a Pandas series with the output dtype set in which case the plan, row\_fn, and init\_state\_fn are not immediately executed.

<p class="api-header">Returns</p>

: __BodoSeries__

<p class="api-header">Example</p>

``` py
import bodo.pandas as pd

def init_state():
    return {1:7}

def per_row(state, row):
    return "bodo" + str(row + state[1])

a = pd.Series(list(range(20)))
b = a.map_with_state(init_state, per_row, output_type=pd.Series(dtype="string[pyarrow]"))
print(b)
```

Output:
```
0      bodo7
1      bodo8
2      bodo9
3     bodo10
4     bodo11
5     bodo12
6     bodo13
7     bodo14
8     bodo15
9     bodo16
10    bodo17
11    bodo18
12    bodo19
13    bodo20
14    bodo21
15    bodo22
16    bodo23
17    bodo24
18    bodo25
19    bodo26
dtype: large_string[pyarrow]
```

---
