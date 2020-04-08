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

Deterministic dataframe schemas, which is required in most data system, is key
for type stability. For example, column `A` of variable `df` in example below could be
either of type integer or string based on a flag -- Bodo cannot determine it at compilation time::

    @bodo.jit
    def f(a):
        if len(a) > 3:  # some computation that cannot be inferred statically
            df = pd.DataFrame({"A": [1, 2, 3]})
        else:
            df = pd.DataFrame({"A": ["a", "b", "c"]})
        return df

    f([2, 3])
    # TypeError: Cannot unify dataframe((array(int64, 1d, C),), RangeIndexType(none), ('A',), False)
    # and dataframe((StringArrayType(),), RangeIndexType(none), ('A',), False)

The error message means that Bodo cannot find a type that can `unify` the two
types into a single type.
This code can be refactored so that `if flag`
is executed in regular Python context, but the rest of computation is in Bodo functions.

Another common place where schema stability may be compromised is in passing non-constant
list of key column names to dataframe operations such as `groupby`, `merge` and `sort_values`.
In these operations, the list of key column names should be constant in order to determine
the output dataframe schema. For example, the program below is potentially type unstable
since Bodo may not be able to infer `column_list` during compilation::

    @bodo.jit
    def f(a):
        column_list = a[0]  # some computation that cannot be inferred statically
        df = pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]})
        return df.groupby(column_list).sum()

    f(["A"])
    # BodoError: groupby(): 'by' parameter only supports a constant column label or column labels.


Bodo supports implicitly inferring constant lists automatically for list addition
and set difference operations such as::

    df.groupby(["A"] + ["B"]).sum()
    df.groupby(list(set(df.columns) - set(["A", "C"]))).sum()

Referring to dataframe columns (e.g. `df["A"]`) requires constants for schema stability as well.
`for` loops over dataframe column names such as below is not supported yet::


    @bodo.jit
    def f(df):
        s = 0
        for c in df.columns:
            s += df[c].sum()
        return s

    f(pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 6]}))
    # BodoError: df[] getitem using unicode_type not supported


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
individual values (i.e. using `[]` with an integer index) may not be type stable::

    @bodo.jit
    def f(S, i):
        return S.iloc[i]  # not type stable
    S = pd.Series(["A", None, "CC"])

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


.. _heterogeneousdtype:

Heterogeneous types inside a data structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :code:`List` containing values of heterogeneous type
	- :code:`myList = [1, "a", 0.1]`
- :code:`Dictionary` containing values of heterogeneous type
	- :code:`myDict = {"A": 1, "B": "a", "C": 0.1}`
