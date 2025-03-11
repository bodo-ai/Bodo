# Performance Measurement {#performance}

This page provides tips on measuring performance of Bodo programs. It
is important to keep the following in mind when measuring program run
time:

1.  Every program has some overhead, so large datasets may be
    necessary for useful measurements.
2.  Performance can vary from one run to another. Several measurements
    are always needed.
3.  Longer computations typically provide more reliable run time
    information.
4.  It is important to use a sequence of tests with increasing input
    size, which helps understand the impact of problem size on program
    performance.
5.  Testing with different data (in terms statistical distribution and
    skew) can be useful to see the impact of data skew on performance
    and scaling.
6.  Simple programs are useful to study performance factors. Complex
    programs are impacted by multiple factors and their performance is
    harder to understand.

## Measuring execution time of Bodo functions

Since Bodo-decorated functions are
[JIT-compiled](https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html){target="blank"},
the compilation time is non-negligible but it only happens the first
time a function is compiled. Compiled functions stay in memory and
don't need to be re-compiled, and they can also be cached to disk (see
[caching][caching]) to be reused across different
executions.

To avoid measuring compilation time, place timers inside the functions.
For example:

```py
"""
calc_pi.py: computes the value of Pi using Monte-Carlo Integration
"""

import numpy as np
import bodo
import time

n = 2 * 10**8

def calc_pi(n):
    t1 = time.time()
    x = 2 * np.random.ranf(n) - 1
    y = 2 * np.random.ranf(n) - 1
    pi = 4 * np.sum(x**2 + y**2 < 1) / n
    print("Execution time:", time.time()-t1, "\nresult:", pi)
    return pi

bodo_calc_pi = bodo.jit(calc_pi)
print("python:")
calc_pi(n)
print("\nbodo:")
bodo_calc_pi(n)
```

The output of this code on a single core is as follows:

```console
python:
Execution time: 5.060443162918091
result: 3.14165914

bodo:
Execution time: 2.165610068012029
result: 3.14154512
```

Bodo's parallel speedup can be measured similarly:

```py
"""
calc_pi.py: computes the value of Pi using Monte-Carlo Integration
"""

import numpy as np
import bodo
import time

@bodo.jit
def calc_pi(n):
    t1 = time.time()
    x = 2 * np.random.ranf(n) - 1
    y = 2 * np.random.ranf(n) - 1
    pi = 4 * np.sum(x**2 + y**2 < 1) / n
    print("Execution time:", time.time()-t1, "\nresult:", pi)
    return pi

calc_pi(2 * 10**8)
```

Launched on eight parallel cores:

```console
$ BODO_NUM_WORKERS=8 python calc_pi.py
Execution time: 0.5736249439651147
result: 3.14161474
```

And the time it takes can be compared with Python performance. Here, we
have a `5.06/0.57 ~= 9x` speedup (from parallelism and sequential
optimizations).

In addition, [SPMD launch mode](../bodo_parallelism/bodo_parallelism_basics.md#spmd) is recommended
for performance measurements since it has lower overheads.


## Measuring sections inside Bodo functions

We can add multiple timers inside a function to see how much time each
section takes:

```py
"""
calc_pi.py: computes the value of Pi using Monte-Carlo Integration
"""

import numpy as np
import bodo
import time

n = 2 * 10**8

def calc_pi(n):
    t1 = time.time()
    x = 2 * np.random.ranf(n) - 1
    y = 2 * np.random.ranf(n) - 1
    t2 = time.time()
    print("Initializing x,y takes: ", t2-t1)

    pi = 4 * np.sum(x**2 + y**2 < 1) / n
    print("calculation takes:", time.time()-t2, "\nresult:", pi)
    return pi

bodo_calc_pi = bodo.jit(calc_pi)
print("python: ------------------")
calc_pi(n)
print("\nbodo: ------------------")
bodo_calc_pi(n)
```

The output is as follows:

```console
python: ------------------
Initializing x,y takes:  3.9832258224487305
calculation takes: 1.1460411548614502
result: 3.14156454

bodo: ------------------
Initializing x,y takes:  3.0611653940286487
calculation takes: 0.35728363902308047
result: 3.14155538
```

!!! note
    Note that Bodo execution took longer in the last example than previous
    ones, since the presence of timers in the middle of computation can
    inhibit some code optimizations (e.g. code reordering and fusion).
    Therefore, one should be cautious about adding timers in the middle of
    computation.


## Disabling JIT Compilation {#disable-jit}

Sometimes it is convenient to disable JIT compilation without removing
the `jit` decorators in the code, to enable easy performance comparison
with regular Python or perform debugging. This can be done by setting
the environment variable `NUMBA_DISABLE_JIT` to `1`, which makes the jit
decorators act as if they perform no operation. In this case, the
invocation of decorated functions calls the original Python functions
instead of compiled versions.

## Load Imbalance

Bodo distributes and processes equal amounts of data across cores as
much as possible. There are certain cases, however, where depending on
the statistical properties of the data and the operation being performed
on it, some cores will need to process much more data than others at
certain points in the application, which limits the scaling that can be
achieved. How much this impacts performance depends on the degree of
imbalance and the impact the affected operation has on overall execution
time.

For example, consider the following operation:

```py
df.groupby("A")["B"].nunique()
```

Where `df` has one billion rows, `A` only has 3 unique values, and we
are running this on a cluster with 1000 cores. Although the work can be
distributed to a certain extent, the final result for each group of `A`
has to be computed on a single core. Because there are only 3 groups,
during computation of the final result there will only be at most three
cores active.

## Expected Scaling

Scaling can be measured as the speedup achieved with *n* cores compared
to running on a single core, that is, the ratio of execution time with 1
core vs *n* cores.

For a fixed input size, the speed up achieved by Bodo with increasing
number of cores (also known as *strong scaling*) depends on a
combination of various factors: size of the input data (problem size),
properties of the data, compute operations used, and the hardware
platform's attributes (such as effective network throughput).

For example, the program above can scale almost linearly (e.g. 100x
speed up on 100 cores) for large enough problem sizes, since the only
communication overhead is parallel summation of the partial sums
obtained by `np.sum` on each processor.

On the other hand, some operations such as join and groupby may require
communicating significant amounts of data across the network, depending
on the characteristics of the data and the exact operation (e.g.
`groupby.sum`, `groupby.nunique`, `groupy.apply`, inner vs outer `join`, etc.),
requiring fast cluster interconnection networks to scale to large number
of cores.

Load imbalance, as described above, can also significantly impair
scaling in certain situations.
