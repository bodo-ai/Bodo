
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
using the @bodo.jit decorator. It replaces the decorated Python functions
with an optimized and parallelized binary version using advanced compilation
methods.
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


This code runs on all environments including laptops, desktops, clusters,
cloud platforms, and edge devices.
Programmers can use :ref:`Jupyter Notebook <jupyter>` or simply
use command line such as `mpiexec -n 1024 python pi.py`.
Bodo is built on top of open source technologies such as
`Numba <https://github.com/numba/numba>`_,
`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_,
and `Apache Arrow <https://arrow.apache.org/>`_.
