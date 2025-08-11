# bodo.pandas.BodoSeries.map\_with\_state
```
BodoSeries.map_partitions_with_state(init_state_fn, func, *args, output_type=None, **kwargs) -> BodoSeries

```
Map a partition of a BodoSeries according to a mapping with a one-time initialization routine whose
result is passed to the partition mapping function.

Sometimes, there is initialization code that is so expensive to run that one would like to minimize
overheads by running it just once per worker.  Other APIs such as map or map\_partitions are
not suitable for this purpose because those would require per row initialization or per partition
initialization.  Likewise, map\_with\_state may not be appropriate if there are significant per-row
overheads that can be amortized across a batch using map\_partitions\_with\_state.

!!! note
    Calling `BodoSeries.map_partitions_with_state` will immediately execute a plan and will perform an
    initialization of the state followed by running func on a small number of rows in order to
    determine the output type of the series.  This plan execution and initialization can be avoided
    if the output\_type is manually specified.

!!! note
    If a per-worker clean-up is required then state can be an instance of a class with the
    \_\_del\_\_ method defined in which the clean-up is performed.

<p class="api-header">Parameters</p>

: __init_state_fn : *function*:__ Initialization function.  Run only once per worker.

: __func : *function*:__ Mapping correspondence.  The first argument to func when called is the previously initialized state variable.  The second argument is the partition to map in the form of a Pandas series.

: __*args : *tuple of Any*:__ Additional positional arguments to func.

: __output_type: *{None, Pandas.series}, default None*:__ If none, then plan is executed and sample of rows passed to func after calling init\_state\_fn to determine the output type.  This parameter can be a Pandas series with the output dtype set in which case the plan, func, and init\_state\_fn are not immediately executed.

: __**kwargs : *dict of string to Any*:__ Additional keyword arguments to func.

<p class="api-header">Returns</p>

: __BodoSeries__

<p class="api-header">Example</p>

``` py
import bodo.pandas as pd

class mystate:
    def __init__(self):
        self.dict = {1:7}

def init_state():
    return mystate()

def per_batch(state, batch, *args, **kwargs):
    def per_row(row):
        return "bodo" + str(row + state.dict[1])
    return batch.map(per_row)

a = pd.Series(list(range(20)))
b = a.map_partitions_with_state(init_state, per_batch, output_type=pd.Series(dtype="string[pyarrow]"))
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
