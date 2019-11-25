Overview
========

Bodo is the simplest and most efficient analytics programming framework.
Bodo Core Engine's advanced technology accelerates and
scales data science programs automatically and enables instant deployment,
eliminating the need to rewrite Python analytics code to Spark/Scala,
SQL or MPI/C++.
This user manual covers the basics of using Bodo, provides a reference
of supported Python features/APIs, and explains how Bodo works under the hoods.

In a nutshell, Bodo provides a just-in-time (JIT) compilation workflow
using the :ref:`@bodo.jit decorator <decorator>`.
It replaces the decorated Python functions
with an optimized and parallelized binary version automatically,
using advanced compilation methods.
For example, the program below computes the value of Pi using Monte-Carlo
Integration::

    @bodo.jit
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x**2 + y**2 < 1) / n
        print("Execution time:", time.time()-t1, "\nresult:", pi)
        return pi

    calc_pi(2 * 10**8)

To run Bodo programs such as this example, programmers can
simply use command line such as `mpiexec -n 1024 python pi.py`,
or use :ref:`Jupyter Notebook <jupyter>`.

Bodo enables scaling and deployment of sequential analytics programs on all
environments including laptops, desktops, clusters, cloud platforms,
and edge devices.
:ref:`Automatic parallelization <supported>` is performed by taking advantage
of Python APIs (Pandas, Numpy, ...) that have data-parallel semantics.
For example, output elements of `x**2` operation in the example above can be
computed independently on different processor nodes and cores.
More complex operations such as join and rolling window
computation can also be parallelized.

The speed up achieved using Bodo depends on various factors such problem size,
parallel overheads of the operations, and hardware platform's attributes.
For example, the program above can scale almost linearly
(e.g. 100 speed on 100 cores)
for large enough problem sizes, since the only communication requirement is
parallel summation for `np.sum`.
On the other hand, workloads with several join and groupby operations
require significant communication of data, requiring fast cluster
interconnection networks to scale to large number of cores.
