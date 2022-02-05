.. _objmode:

Using Regular Python inside JIT (Object Mode)
=============================================


Regular Python functions and Bodo JIT functions can be used together in applications arbitrarily,
but there are cases where regular Python code needs to be used *inside* JIT code.
For example, you may want to use Bodo's parallel constructs with some code that does not have JIT
support yet.
*Object Mode* allows switching to a Python interpreted context to be able
to run non-jittable code.
The main requirement is that the user has to
specify the type of variables used in later JIT code.

.. The name *Object Mode* means that all variables will be regular
.. Python objects instead of native binary JIT values.


For example, the following code calls a non-JIT function
on rows of a distributed dataframe.
The code inside ``with bodo.objmode`` runs as regular Python, but variable ``y``
is returned to JIT code (since it is used after the ``with`` block).
Therefore, the ``y="float64"`` type annotation is required.


.. code:: ipython3

    import pandas as pd
    import numpy as np
    import bodo
    import scipy.special as sc


    def my_non_jit_function(a, b):
        return np.log(a) + sc.entr(b)


    @bodo.jit
    def f(row):
        with bodo.objmode(y="float64"):
            y = my_non_jit_function(row.A, row.B)
        return y


    @bodo.jit
    def objmode_example(n):
        df = pd.DataFrame({"A": np.random.ranf(n), "B": np.arange(n)})
        df["C"] = df.apply(f, axis=1)
        print(df["C"].sum())

    objmode_example(10)


We recommend keeping the code inside the ``with bodo.objmode`` block minimal and call
outside Python functions instead (as in this example).
This reduces compilation time and sidesteps potential compiler limitations.


Object Mode Type Annotations
----------------------------

There are various ways to specify the data types in ``objmode``.
Basic data types such as ``float64`` and ``int64`` can be specified as string
values (as in the previous example).
For more complex data types like dataframes, ``bodo.typeof()`` can be used on sample data
that has the same type as expected outputs. For example:

.. code:: ipython3

    df_sample = pd.DataFrame({"A": [0], "B": ["AB"]}, index=[0])
    df_type = bodo.typeof(df_sample)

    @bodo.jit
    def f():
        with bodo.objmode(df=df_type):
            df = pd.DataFrame({"A": [1, 2, 3], "B": ["ab", "bc", "cd"]}, index=[3, 2, 1])
        return df


This is equivalent to creating the ``DataFrameType`` directly:

.. code:: ipython3

    @bodo.jit
    def f():
        with bodo.objmode(
            df=bodo.DataFrameType(
                (bodo.int64[::1], bodo.string_array_type),
                bodo.NumericIndexType(bodo.int64),
                ("A", "B"),
            )
        ):
            df = pd.DataFrame({"A": [1, 2, 3], "B": ["ab", "bc", "cd"]}, index=[3, 2, 1])
        return df


The data type can be registered in Bodo so it can be referenced using a string name later:

.. code:: ipython3

    df_sample = pd.DataFrame({"A": [0], "B": ["AB"]}, index=[0])
    bodo.register_type("my_df_type", bodo.typeof(df_sample))

    @bodo.jit
    def f():
        with bodo.objmode(df="my_df_type"):
            df = pd.DataFrame({"A": [1, 2, 3], "B": ["ab", "bc", "cd"]}, index=[3, 2, 1])
        return df


See :ref:`pandas-dtype` for more details on Bodo data types in general.
Bodo's Object Mode is built on top of Numba's Object Mode (see Numba `objmode <http://numba.pydata.org/numba-doc/latest/user/withobjmode.html#the-objmode-context-manager>`__
for more details).



What Can Be Done Inside Object Mode
-----------------------------------

The code inside Object Mode runs in regular Python on all parallel processes,
which means Object Mode does not include Bodo compiler's automatic parallel communication management.
Therefore, the computation inside Object Mode should be
independent on different processors and not require communication. In general:

- Operations on scalars are safe
- Operations that compute on rows independently are safe
- Operations that compute across rows may not be safe


The example below demonstrates a valid use of Object Mode,
since it uses `df.apply(axis=1)` which runs on different rows independently.

.. code:: ipython3

    df_type = bodo.typeof(pd.DataFrame({"A": [1], "B": [1], "C": [1]}))

    def f(df):
        return df.assign(C=df.apply(lambda r: r.A + r.B, axis=1))

    @bodo.jit
    def valid_objmode():
        df = pd.read_parquet("in_file.pq")
        with bodo.objmode(df2=df_type):
            df2 = f(df)
        df2.to_parquet("out_file.pq")

    valid_objmode()

In contrast, the example below demonstrates an invalid use of Object Mode.
The reason is that groupby computation requires grouping together all rows
with the same key across all chunks.
However, on each processor, Bodo passes a chunk of `df` to Object Mode
which returns results from local groupby computation.
Therefore, `df2` does not include valid global groupby output.

.. code:: ipython3

    df_type = bodo.typeof(pd.DataFrame({"A": [1], "B": [1]}))

    def f(df):
        return df.groupby("A", as_index=False).sum()

    @bodo.jit
    def invalid_objmode():
        df = pd.read_parquet("in_file.pq")
        # Invalid use of objmode
        with bodo.objmode(df2=df_type):
            df2 = f(df)
        df2.to_parquet("out_file.pq")

    invalid_objmode()


Groupby/Apply Object Mode Pattern
---------------------------------

ML algorithms and other complex data science computations are often called
on groups of dataframe rows.
Bodo supports parallelizing these computations (which may not have JIT support yet)
using Object Mode inside ``groupby/apply``.
For example, the code below runs `Prophet <https://facebook.github.io/prophet/>`_
on groups of rows.
This is a valid use of Object Mode since Bodo handles shuffle communication for groupby/apply and
brings all rows of each group in the same local chunk.
Therefore, the apply function running in Object Mode has all the data it needs.

.. code:: ipython3

    import bodo
    import pandas as pd
    import numpy as np
    from fbprophet import Prophet

    prophet_output_type = bodo.typeof(pd.DataFrame({"ds": pd.date_range("2017-01-03", periods=1), "yhat": [0.0]}))

    def run_prophet(df):
        m = Prophet()
        m.fit(df)
        return m.predict(df)[["ds", "yhat"]]

    @bodo.jit
    def apply_func(df):
        with bodo.objmode(df2=prophet_output_type):
            df2 = run_prophet(df)
        return df2

    @bodo.jit
    def f(df):
        df2 = df.groupby("A").apply(apply_func)
        return df2

    n = 10
    df = pd.DataFrame({"A": np.arange(n) % 3, "ds": pd.date_range("2017-01-03", periods=n), "y": np.arange(n)})
    print(f(df))


