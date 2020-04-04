.. _notsupported:

Unsupported Python 
=============================

Bodo statically compiles user codes to generate efficient parallel programs.
Hence, the user code needs to be `statically compilable`.
This means that Bodo should be able to infer all the variable types, and be able
to analyze the computations.

.. _typestability:

Type Stability
--------------

To enable type inference, the program should be `type stable`, which means every
variable should have a single type. The example below is not type stable since
variable ``a`` can be both a float and an array of floats::

    if flag:
        a = 1.0
    else:
        a = np.ones(10)

The use of ``isinstance`` operator of Python often means type instability and
is not supported.

Similarly, function calls should also be deterministic. The below example is
not supported since function ``f`` is not known in advance::

    if flag:
        f = np.zeros
    else:
        f = np.random.ranf
    A = f(10)

One can usually avoid these cases in numerical code without significant effort.

Type stability forces us to change the behavior of some functions. For example
we have here a difference of behavior::

    def test_impl(df):
        return df["A"].iat[1]
    df1 = pd.DataFrame({"A":[["A"], np.nan, ["AB", "CD"]]})
    bodo_impl = bodo.jit(test_impl)
    df2_pandas = test_impl(df1) # Will be nan
    df2_bodo = bodo_impl(df1)   # Will be []

This difference in behavior is needed in order to enforce the type stability of the
getitem function.

.. _heterogeneousdtype:

Heterogeneous types inside a data structure
---------------------------------------------

- :code:`List` containing values of heterogeneous type
	- :code:`myList = [1, "a", 0.1]`
- :code:`Dictionary` containing values of heterogeneous type
	- :code:`myDict = {"A": 1, "B": "a", "C": 0.1}`
