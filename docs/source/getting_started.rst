Getting Started
===============

Bodo provides the `@bodo.jit` decorator which automatically optimizes and parallelizes
analytics programs.
For example, the following program creates a distributed data frame and prints the first
three rows::

    import numpy as np
    import pandas as pd
    import bodo

    @bodo.jit
    def f(n):
        return pd.DataFrame({'A': np.arange(n)}).head(3)

    print(f(100))


This Bodo program can be launched on four parallel cores as follows::

    $ mpiexec -n 4 python example.py
       A
    0  0
    1  1
    2  2
       A
    0  0
    1  1
    2  2
       A
    0  0
    1  1
    2  2
       A
    0  0
    1  1
    2  2


In this example, each core owns a chunk of the distributed data frame (25 rows in this case).
However, output of `head` is not distributed
and will be available on all processors.
