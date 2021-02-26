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

Load Balancing Distributed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some computations such as ``filter``, ``join`` or ``groupby`` can result in imbalanced data chunks across cores for distributed data.
This may result in some cores operating on nearly empty dataframes, and others on relatively large ones.

Bodo provides ``bodo.rebalance`` to allow manual load balance if necessary. For example::


    @bodo.jit(distributed={"df"})
    def rebalance_example(df):
        df = df[df["A"] > 3]
        df = bodo.rebalance(df)
        return df.sum()

In this case, we use `bodo.rebalance` to make sure the filtered dataframe has near-equal data chunk sizes across cores, which would accelerate later computations (`sum` in this case).


We can also use the `dests` keyword to specify a subset of ranks to which bodo should distribute the data from all ranks.

Example usage::

    @bodo.jit(distributed={"df"})
    def rebalance_example(df):
        df = df[df["A"] > 3]
        df = bodo.rebalance(df, dests=[0, 1])
        return df.sum()
