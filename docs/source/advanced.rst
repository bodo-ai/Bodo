.. _advanced:


Advanced Parallelism Topics
---------------------------

This section discusses parallelism topics that may be useful for performance tuning and
advanced use cases.

Getting/Setting Distributed Data Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributed data is usually accessed and modified through high-level Pandas and Numpy
APIs. However, Bodo allows direct access to distributed data without code modification in many cases as well.
Here are the cases that are currently supported:

#. Getting values using boolean array indexing, e.g. `B = A[A > 3]`.
   The output can be distributed, but may be imbalanced (`bodo.rebalance()` can be used if necessary).
#. Getting values using a slice, e.g. `B = A[::2]`.
   The output can be distributed, but may be imbalanced (`bodo.rebalance()` can be used if necessary).
#. Getting a value using a scalar index, e.g. `a = A[m]`.
   The output can be replicated.

#. Setting values using boolean array indexing, e.g. `A[A > 3] = a`.
   Only supports setting a scalar or lower-dimension value currently.
#. Setting values using a slice, e.g. `A[::2] = a`.
   Only supports setting a scalar or lower-dimension value currently.
#. Setting a value using a scalar index, e.g. `A[m] = a`.
