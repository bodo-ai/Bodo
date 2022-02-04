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

#. Getting values using boolean array indexing, e.g. ``B = A[A > 3]``.
   The output can be distributed, but may be imbalanced (``bodo.rebalance()`` can be used if necessary).
#. Getting values using a slice, e.g. ``B = A[::2]``.
   The output can be distributed, but may be imbalanced (``bodo.rebalance()`` can be used if necessary).
#. Getting a value using a scalar index, e.g. ``a = A[m]``.
   The output can be replicated.

#. Setting values using boolean array indexing, e.g. ``A[A > 3] = a``.
   Only supports setting a scalar or lower-dimension value currently.
#. Setting values using a slice, e.g. ``A[::2] = a``.
   Only supports setting a scalar or lower-dimension value currently.
#. Setting a value using a scalar index, e.g. ``A[m] = a``.


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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some computations such as ``filter``, ``join`` or ``groupby`` can result in imbalanced data chunks across cores for distributed data.
This may result in some cores operating on nearly empty dataframes, and others on relatively large ones.

.. TODO: Add link to rebalance API 

Bodo provides ``bodo.rebalance`` to allow manual load balance if necessary. For example::


    @bodo.jit(distributed={"df"})
    def rebalance_example(df):
        df = df[df["A"] > 3]
        df = bodo.rebalance(df)
        return df.sum()


In this case, we use ``bodo.rebalance`` to make sure the filtered dataframe has near-equal data chunk sizes across cores, which would accelerate later computations (``sum`` in this case).


We can also use the ``dests`` keyword to specify a subset of ranks to which bodo should distribute the data from all ranks.

Example usage::

    @bodo.jit(distributed={"df"})
    def rebalance_example(df):
        df = df[df["A"] > 3]
        df = bodo.rebalance(df, dests=[0, 1])
        return df.sum()


Explicit Parallel Loops
~~~~~~~~~~~~~~~~~~~~~~~

Sometimes explicit parallel loops are required since a program cannot be
written in terms of data-parallel operators easily. In this case, one
can use Bodo’s ``prange`` in place of ``range`` to specify that a loop
can be parallelized. The user is required to make sure the loop does not
have cross iteration dependencies except for supported reductions.

The example below demonstrates a parallel loop with a reduction:

.. code:: ipython3

    import bodo
    from bodo import prange
    import numpy as np

    @bodo.jit
    def prange_test(n):
        A = np.random.ranf(n)
        s = 0
        B = np.empty(n)
        for i in prange(len(A)):
            bodo.parallel_print("rank", bodo.get_rank())
            # A[i]: distributed data access with loop index
            # s: a supported sum reduction
            s += A[i]
            # write array with loop index
            B[i] = 2 * A[i]
        return s + B.sum()

    res = prange_test(10)
    print(res)


.. parsed-literal::

    [stdout:0]
    rank 0
    rank 0
    rank 0
    13.077183553245497
    [stdout:1]
    rank 1
    rank 1
    rank 1
    13.077183553245497
    [stdout:2]
    rank 2
    rank 2
    13.077183553245497
    [stdout:3]
    rank 3
    rank 3
    13.077183553245497


Currently, reductions using +=, \*=, min, and max operators are
supported. Iterations are simply divided between processes and executed
in parallel, but reductions are handled using data exchange.

Integration with non-Bodo APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are multiple methods for integration with APIs that Bodo does not
support natively: 1. Switch to python object mode inside jit functions
2. Pass data in and out of jit functions


Passing Distributed Data
^^^^^^^^^^^^^^^^^^^^^^^^

Bodo can receive or return chunks of distributed data to allow flexible
integration with any non-Bodo Python code. The following example passes
chunks of data to interpolate with Scipy, and returns interpolation
results back to jit function.

.. code:: ipython3

    import scipy.interpolate

    @bodo.jit(distributed=["X", "Y", "X2"])
    def dist_pass_test(n):
        X = np.arange(n)
        Y = np.exp(-X/3.0)
        X2 = np.arange(0, n, 0.5)
        return X, Y, X2

    X, Y, X2 = dist_pass_test(100)
    # clip potential out-of-range values
    X2 = np.minimum(np.maximum(X2, X[0]), X[-1])
    f = scipy.interpolate.interp1d(X, Y)
    Y2 = f(X2)

    @bodo.jit(distributed={"Y2"})
    def dist_pass_res(Y2):
        return Y2.sum()

    res = dist_pass_res(Y2)
    print(res)


.. parsed-literal::

    [stdout:0] 6.555500504321469
    [stdout:1] 6.555500504321469
    [stdout:2] 6.555500504321469
    [stdout:3] 6.555500504321469


Collections of Distributed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List and dictionary collections can be used to hold distributed data
structures:

.. code:: ipython3


    @bodo.jit(distributed=["df"])
    def f():
        to_concat = []
        for i in range(10):
            to_concat.append(pd.DataFrame({'A': np.arange(100), 'B': np.random.random(100)}))
            df = pd.concat(to_concat)
        return df

    f()



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0.518256</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>0.996147</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>0.881703</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>0.821504</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>0.311216</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>20</th>
          <td>20</td>
          <td>0.440666</td>
        </tr>
        <tr>
          <th>21</th>
          <td>21</td>
          <td>0.142903</td>
        </tr>
        <tr>
          <th>22</th>
          <td>22</td>
          <td>0.825534</td>
        </tr>
        <tr>
          <th>23</th>
          <td>23</td>
          <td>0.359685</td>
        </tr>
        <tr>
          <th>24</th>
          <td>24</td>
          <td>0.534700</td>
        </tr>
      </tbody>
    </table>
    <p>250 rows × 2 columns</p>
    </div>



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>25</th>
          <td>25</td>
          <td>0.284761</td>
        </tr>
        <tr>
          <th>26</th>
          <td>26</td>
          <td>0.441711</td>
        </tr>
        <tr>
          <th>27</th>
          <td>27</td>
          <td>0.468827</td>
        </tr>
        <tr>
          <th>28</th>
          <td>28</td>
          <td>0.015361</td>
        </tr>
        <tr>
          <th>29</th>
          <td>29</td>
          <td>0.002683</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>45</th>
          <td>45</td>
          <td>0.217445</td>
        </tr>
        <tr>
          <th>46</th>
          <td>46</td>
          <td>0.372188</td>
        </tr>
        <tr>
          <th>47</th>
          <td>47</td>
          <td>0.737716</td>
        </tr>
        <tr>
          <th>48</th>
          <td>48</td>
          <td>0.168481</td>
        </tr>
        <tr>
          <th>49</th>
          <td>49</td>
          <td>0.757296</td>
        </tr>
      </tbody>
    </table>
    <p>250 rows × 2 columns</p>
    </div>



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>50</th>
          <td>50</td>
          <td>0.430431</td>
        </tr>
        <tr>
          <th>51</th>
          <td>51</td>
          <td>0.572574</td>
        </tr>
        <tr>
          <th>52</th>
          <td>52</td>
          <td>0.347954</td>
        </tr>
        <tr>
          <th>53</th>
          <td>53</td>
          <td>0.547276</td>
        </tr>
        <tr>
          <th>54</th>
          <td>54</td>
          <td>0.558948</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>70</th>
          <td>70</td>
          <td>0.768203</td>
        </tr>
        <tr>
          <th>71</th>
          <td>71</td>
          <td>0.106369</td>
        </tr>
        <tr>
          <th>72</th>
          <td>72</td>
          <td>0.036671</td>
        </tr>
        <tr>
          <th>73</th>
          <td>73</td>
          <td>0.485589</td>
        </tr>
        <tr>
          <th>74</th>
          <td>74</td>
          <td>0.137820</td>
        </tr>
      </tbody>
    </table>
    <p>250 rows × 2 columns</p>
    </div>



.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A</th>
          <th>B</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>75</th>
          <td>75</td>
          <td>0.323295</td>
        </tr>
        <tr>
          <th>76</th>
          <td>76</td>
          <td>0.928662</td>
        </tr>
        <tr>
          <th>77</th>
          <td>77</td>
          <td>0.769746</td>
        </tr>
        <tr>
          <th>78</th>
          <td>78</td>
          <td>0.988702</td>
        </tr>
        <tr>
          <th>79</th>
          <td>79</td>
          <td>0.452371</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>95</th>
          <td>95</td>
          <td>0.458132</td>
        </tr>
        <tr>
          <th>96</th>
          <td>96</td>
          <td>0.959298</td>
        </tr>
        <tr>
          <th>97</th>
          <td>97</td>
          <td>0.988239</td>
        </tr>
        <tr>
          <th>98</th>
          <td>98</td>
          <td>0.797115</td>
        </tr>
        <tr>
          <th>99</th>
          <td>99</td>
          <td>0.071809</td>
        </tr>
      </tbody>
    </table>
    <p>250 rows × 2 columns</p>
    </div>


