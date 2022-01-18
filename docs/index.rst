.. Bodo documentation master file, created by
   sphinx-quickstart on Wed Sep  6 09:29:19 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bodo
====

Bodo is a new just-in-time (JIT) inferential compiler that
brings supercomputing-style performance
and scalability to native Python analytics code automatically.
Bodo has several advantages over other big data analytics systems
(which are usually distributed scheduler libraries):

- Simple programming with native Python APIs like Pandas and Numpy (no "Pandas-like" API layers)
- Extreme performance and scalability using true parallelism and advanced compiler technology
- Very high reliability due to binary code generation, which avoids distributed library failures
- Simple deployment using standard Python workflows
- Flexible integration with other systems such as cloud storage, data warehouses, and visualization tools


.. TODO: Conceptual diagram


This documentation covers the basics of using Bodo and provides a reference
of supported Python features and APIs.
In a nutshell, Bodo provides a JIT compilation workflow
using the :ref:`@bodo.jit decorator <jit>`.
It replaces the decorated Python functions
with an optimized and parallelized binary version automatically.
For example, the program below can perform data transformation on large datasets::

    @bodo.jit
    def data_transform(file_name):
        df = pd.read_parquet(file_name)
        df = df[df.C.dt.month == 1]
        df2 = df.groupby("A")["B", "D"].agg(
            lambda S: (S == "ABC").sum()
        )
        df2.to_parquet("output.pq")


To run Bodo programs such as this example, programmers can
simply use the command line such as `mpiexec -n 1024 python data_transform.py`
(to run on 1024 cores),
or use :ref:`Jupyter Notebook <jupyter>`.


.. panels::

   .. toctree::
      :maxdepth: 1

      source/getting_started

   ---

   .. toctree::
      :maxdepth: 1

      source/installation_and_setup/index

   ---

   .. toctree::
      :maxdepth: 1

      source/programming_with_bodo/index

   ---

   .. toctree::
      :maxdepth: 1

      source/Integrating_bodo/index

   ---

   .. toctree::
      :maxdepth: 1

      source/performance_and_diagnostics/index

   ---

   .. toctree::
      :maxdepth: 1

      source/help_and_reference/index