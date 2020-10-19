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


Concatenation Reduction
~~~~~~~~~~~~~~~~~~~~~~~

Some algorithms require generating variable-length output data per input
data element. Bodo supports parallelizing this pattern, which we refer to as
`concatenation reduction`. For example::

   @bodo.jit
   def impl(n):
      df = pd.DataFrame()
      for i in bodo.prange(n):
         df = df.append(pd.DataFrame({"A": np.arange(i)}))

      return df

A common use case is simulation applications that generate possible outcomes based on parameters.
For example::

   @bodo.jit
   def impl():
      params = np.array([0.1, 0.2, 0.5, 1.0, 1.2, 1.5, ..., 100])
      params = bodo.scatterv(params)
      df = pd.DataFrame()
      for i in bodo.prange(len(params)):
         df = df.append(get_result(params[i]))

      return df

In this example, we chose to manually parallelize the parameter array for simplicity, since the workload
is compute-heavy and the parameter data is relatively small.
