.. Bodo documentation master file, created by
   sphinx-quickstart on Wed Sep  6 09:29:19 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bodo
====


Bodo is a new compute engine that brings supercomputing-like performance
and scalability to native Python analytics code automatically.
With Bodo, Python/Pandas code scales to 10,000+ cores and petabytes of
data without any rewrites and the inefficiencies
and frequent failures of distributed libraries.
Bodo does this using a new just-in-time (JIT) inferential compiler technology
that can automate the optimization process of world-class performance experts.


This documentation covers the basics of using Bodo and provides a reference
of supported Python features and APIs.
In a nutshell, Bodo provides a JIT compilation workflow
using the `@bodo.jit decorator <user_guide.html#jit-just-in-time-compilation-workflow>`__.
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

Automatic parallelization is performed by taking advantage
of Python APIs (Pandas, Numpy, ...) that have data-parallel semantics.
For example, output elements of `df.C.dt.month` operation in the example above can be
computed independently on different processor nodes and cores.
More complex operations such as groupby
computation can also be parallelized with some communication across processors.



.. toctree::
   :maxdepth: 2
   :caption: Installation and Setup

   source/install
   source/enterprise
   source/ipyparallel
   source/bodo_platform_aws
   source/bodo_platform_azure
   source/bodo_platform



.. toctree::
   :maxdepth: 2
   :caption: Programming with Bodo

   source/getting_started
   source/user_guide
   source/file_io
   source/pandas
   source/numpy
   source/ml
   source/dl
   source/data_visualization
   source/not_supported
   source/advanced
   source/BodoSQL
   source/e3

.. toctree::
   :maxdepth: 2
   :caption: Migration from Spark

   source/spark
   source/sparkcheatsheet


.. toctree::
   :maxdepth: 2
   :caption: Performance and Diagnostics

   source/caching
   source/performance
   source/compilation
   source/troubleaws
   source/troubleazure

.. toctree::
   :maxdepth: 2
   :caption: Help and Reference

   source/releases
   source/eula
   source/prev_doc_link
