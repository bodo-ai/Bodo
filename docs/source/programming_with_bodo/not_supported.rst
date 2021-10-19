.. _notsupported:

Unsupported Python Programs
===========================

Bodo compiles functions into efficient native parallel binaries, which
requires all the operations used in the code to be
supported by Bodo. This excludes some Python features explained in this
section.


.. _typestability:

Type Stability
--------------

To enable type inference, the program should be `type stable`, which means Bodo
should be able to assign a single type to every variable.


DataFrame Schema
~~~~~~~~~~~~~~~~

Deterministic dataframe schemas, which are required in most data systems, is key
for type stability. For example, variable `df` in example below could be
either a single column dataframe or a two column one -- Bodo cannot determine it at compilation time::

    @bodo.jit
    def f(a):
        df = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"A": [1, 3, 4], "C": [-1, -2, -3]})
        if len(a) > 3:
            df = df.merge(df2)

        return df.mean()

    print(f([2, 3]))
    # TypeError: Cannot unify dataframe((array(int64, 1d, C),), RangeIndexType(none), ('A',), False)
    # and dataframe((array(int64, 1d, C), array(int64, 1d, C)), RangeIndexType(none), ('A', 'C'), False) for 'df'

The error message means that Bodo cannot find a type that can `unify` the two
types into a single type.
This code can be refactored so that the `if` control flow
is executed in regular Python context, but the rest of computation is in Bodo functions.
For example, one could use two versions of the function::

    @bodo.jit
    def f1():
        df = pd.DataFrame({"A": [1, 2, 3]})
        return df.mean()

    @bodo.jit
    def f2():
        df = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"A": [1, 3, 4], "C": [-1, -2, -3]})
        df = df.merge(df2)
        return df.mean()

    a = [2, 3]
    if len(a) > 3:
        print(f1())
    else:
        print(f2())


Another common place where schema stability may be compromised is in passing non-constant
list of key column names to dataframe operations such as `groupby`, `merge` and `sort_values`.
In these operations, Bodo should be able to deduce the list of key column names at compile time
in order to determine the output dataframe schema. For example, the program below is potentially type unstable
since Bodo may not be able to infer `column_list` during compilation::

    @bodo.jit
    def f(a, i):
        column_list = a[:i]  # some computation that cannot be inferred statically
        df = pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]})
        return df.groupby(column_list).sum()

    a = ["A", "B"]
    i = 1
    f(a, i)
    # BodoError: groupby(): 'by' parameter only supports a constant column label or column labels.

This code can be refactored so that the computation for `column_list` is performed
in regular Python context, and the result is passed as a function argument::

    @bodo.jit
    def f(column_list):
        df = pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]})
        return df.groupby(column_list).sum()

    a = ["A", "B"]
    i = 1
    column_list = a[:i]
    f(column_list)

In general, Bodo can infer constants from function arguments, global variables, and
constant values in the program. Furthermore,
Bodo supports implicitly inferring constant lists automatically for list addition
and set difference operations such as::

    df.groupby(["A"] + ["B"]).sum()
    df.groupby(list(set(df.columns) - set(["A", "C"]))).sum()

Bodo will support inferring more implicit constant cases in the future
(e.g. more list and set operations).

Referring to dataframe columns (e.g. `df["A"]`) requires constants for schema stability as well.
`for` loops over dataframe column names such as below is not supported yet::


    @bodo.jit
    def f(df):
        s = 0
        for c in df.columns:
            s += df[c].sum()
        return s

    f(pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]}))
    # BodoError: df[] getitem selecting a subset of columns requires providing constant column names. For more information, see https://docs.bodo.ai/latest/source/programming_with_bodo/require_constants.html


Variable Types and Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example below is not type stable since
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

One can usually avoid these cases in analytics codes without significant effort.


Accessing individual values of nullable data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The type of null (NA) value for most nullable data arrays is different than
regular values (except float data which stores `np.nan`). Therefore, accessing
individual values (i.e. using `[]` with an integer index) may not be type stable.
In these cases, Bodo assumes the value is not NA and returns an "neutral" value::

    @bodo.jit
    def f(S, i):
        return S.iloc[i]  # not type stable
    S = pd.Series(["A", None, "CC"])
    f(S, 1)  # returns ""

The solution is to check for NA values using `pd.isna` to handle NA values appropriately::

    @bodo.jit
    def f(S, i):
        if pd.isna(S.iloc[i]):
            return "NA"
        return S.iloc[i]
    S = pd.Series(["A", None, "CC"])
    f(S, 1)  # returns "NA"

We are working on making it possible to avoid stability issues automatically
in most practical cases.


Unsupported Python Constructs
-----------------------------

Bodo relies on Numba for supporting basic Python features.
Therefore, Python constructs that are not supported by Numba
(see Numba documentation `here <http://numba.pydata.org/numba-doc/latest/reference/pysupported.html>`_)
should be avoided in Bodo programs.

Generally, these Python features are not supported:

* exceptions: `try .. except`, `raise`
* context manager: `with`
* list, set, dict and generator comprehensions
* async features
* class definition: `class`
* jit functions cannot have `**kwargs`
* functions can be passed as arguments but not returned
* lists of lists cannot be passed as arguments unless if
  `typed-list of Numba <http://numba.pydata.org/numba-doc/latest/reference/pysupported.html#typed-list>`_ is used.
* `typed-dict of Numba <http://numba.pydata.org/numba-doc/latest/reference/pysupported.html#typed-dict>`_
  is currently required for passing dictionaries as argument to jit functions.

.. _heterogeneousdtype:

Heterogeneous types inside a data structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :code:`List` containing values of heterogeneous type
	- :code:`myList = [1, "a", 0.1]`
- :code:`Dictionary` containing values of heterogeneous type
	- :code:`myDict = {"A": 1, "B": "a", "C": 0.1}`
