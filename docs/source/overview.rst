Overview
========

Bodo is the first Python Supercomputing Analytics platform that empowers data scientists to run
Python workloads with the extreme performance and scalability of
High Performance Computing (HPC) without code rewrites.

Bodo’s technology provides automatic parallelization and acceleration for analytics workloads
for the first time. This eliminates the need to rewrite Python code in Spark/Scala,
SQL or MPI/C++,
allowing data scientists to focus on solving business problems instead
of scalability and performance of their codes.


This user manual covers the basics of using Bodo, provides a reference
of supported Python features/APIs, and explains how Bodo works under the hoods.
In a nutshell, Bodo provides a just-in-time (JIT) compilation workflow
using the `@bodo.jit decorator <user_guide.html#jit-just-in-time-compilation-workflow>`__.
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

Automatic parallelization is performed by taking advantage
of Python APIs (Pandas, Numpy, ...) that have data-parallel semantics.
For example, output elements of `x**2` operation in the example above can be
computed independently on different processor nodes and cores.
More complex operations such as join and rolling window
computation can also be parallelized.
