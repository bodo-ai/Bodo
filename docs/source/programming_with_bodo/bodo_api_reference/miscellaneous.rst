Miscellaneous Supported Python API
===================================

In this section, we will discuss some useful Bodo features.



Nullable Integers in Pandas
---------------------------

DataFrame and Series objects with integer data need special care due to
`integer NA issues in
Pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions>`__.
By default, Pandas dynamically converts integer columns to floating
point when missing values (NAs) are needed, which can result in loss of
precision as well as type instability.

Pandas introduced `a new nullable integer data
type <https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html#integer-na>`__
that can solve this issue, which is also supported by Bodo. For example,
this code reads column A into a nullable integer array (the capital “I”
denotes nullable integer type):

.. code:: ipython3

    data = (
        "11,1.2\n"
        "-2,\n"
        ",3.1\n"
        "4,-0.1\n"
    )
    
    with open("data/data.csv", "w") as f:
        f.write(data)
    
    
    @bodo.jit(distributed=["df"])
    def f():
        dtype = {"A": "Int64", "B": "float64"}
        df = pd.read_csv("data/data.csv", dtype = dtype, names = dtype.keys())
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
          <td>11</td>
          <td>1.2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-2</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>&lt;NA&gt;</td>
          <td>3.1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>-0.1</td>
        </tr>
      </tbody>
    </table>
    </div>



Checking NA Values
------------------

When an operation iterates over the values in a Series or Array, type
stablity requires special handling for NAs using ``pd.isna()``. For
example, ``Series.map()`` applies an operation to each element in the
series and failing to check for NAs can result in garbage values
propagating.

.. code:: ipython3

    S = pd.Series(pd.array([1, None, None, 3, 10], dtype="Int8"))
    
    @bodo.jit
    def map_copy(S):
        return S.map(lambda a: a if not pd.isna(a) else None)
    
    print(map_copy(S))


.. parsed-literal::

    0       1
    1    <NA>
    2    <NA>
    3       3
    4      10
    dtype: Int8


Boxing/Unboxing Overheads
~~~~~~~~~~~~~~~~~~~~~~~~~

Bodo uses efficient native data structures which can be different than
Python. When Python values are passed to Bodo, they are *unboxed* to
native representation. On the other hand, returning Bodo values requires
*boxing* to Python objects. Boxing and unboxing can have significant
overhead depending on size and type of data. For example, passing string
column between Python/Bodo repeatedly can be expensive:

.. code:: ipython3

    @bodo.jit(distributed=["df"])
    def gen_data():
        df = pd.read_parquet("data/cycling_dataset.pq")
        df["hr"] = df["hr"].astype(str)
        return df
    
    @bodo.jit(distributed=["df", "x"])
    def mean_power(df):
        x = df.hr.str[1:]
        return x
    
    df = gen_data()
    res = mean_power(df)
    print(res)


.. parsed-literal::

    0        1
    1        2
    2        2
    3        3
    4        3
            ..
    3897    00
    3898    00
    3899    00
    3900    00
    3901    00
    Name: hr, Length: 3902, dtype: object


One can try to keep data in Bodo functions as much as possible to avoid
boxing/unboxing overheads:

.. code:: ipython3

    @bodo.jit(distributed=["df"])
    def gen_data():
        df = pd.read_parquet("data/cycling_dataset.pq")
        df["hr"] = df["hr"].astype(str)
        return df
    
    @bodo.jit(distributed=["df", "x"])
    def mean_power(df):
        x = df.hr.str[1:]
        return x
    
    @bodo.jit
    def f():
        df = gen_data()
        res = mean_power(df)
        print(res)
    
    f()


.. parsed-literal::

    0        1
    1        2
    2        2
    3        3
    4        3
            ..
    3897    00
    3898    00
    3899    00
    3900    00
    3901    00
    Name: hr, Length: 3902, dtype: object


Iterating Over Columns
~~~~~~~~~~~~~~~~~~~~~~

Iterating over columns in a dataframe can cause type stability issues,
since column types in each iteration can be different. Bodo supports
this usage for many practical cases by automatically unrolling loops
over dataframe columns when possible. For example, the example below
computes the sum of all data frame columns:

.. code:: ipython3

    @bodo.jit
    def f():
        n = 20
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n) ** 2, "C": np.ones(n)})
        s = 0
        for c in df.columns:
         s += df[c].sum()
        return s
    
    f()




.. parsed-literal::

    2680.0



For automatic unrolling, the loop needs to be a ``for`` loop over column
names that can be determined by Bodo at compile time.

Regular Expressions using ``re``
--------------------------------

Bodo supports string processing using Pandas and the ``re`` standard
package, offering significant flexibility for string processing
applications. For example:

.. code:: ipython3

    import re
    
    @bodo.jit
    def f(S):
        def g(a):
            res = 0
            if re.search(".*AB.*", a):
                res = 3
            if re.search(".*23.*", a):
                res = 5
            return res
    
        return S.map(g)
    
    S = pd.Series(["AABCDE", "BBABCE", "1234"])
    f(S)


.. parsed-literal::

    /Users/user/dev/bodo/bodo/transforms/distributed_analysis.py:229: BodoWarning: No parallelism found for function 'f'. This could be due to unsupported usage. See distributed diagnostics for more information.
      warnings.warn(



.. parsed-literal::

    0    3
    1    3
    2    5
    dtype: int64


Class Support using ``@jitclass``
---------------------------------

Bodo supports Python classes using the @bodo.jitclass decorator. It
requires type annotation of the fields, as well as distributed
annotation where applicable. For example, the example class below holds
a distributed dataframe of values and a name filed. Types can either be
specified directly using the imports in the bodo package or can be
inferred from existing types using ``bodo.typeof``.

.. code:: ipython3

    @bodo.jitclass(
        {
            "df": bodo.DataFrameType(
                    (bodo.int64[::1], bodo.float64[::1]),
                    bodo.RangeIndexType(bodo.none),
                    ("A", "B"),
                ),
            "name": bodo.typeof("hello world"),
        },
        distributed=["df"],
    )
    class BodoClass:
        def __init__(self, n, name):
            self.df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
            self.name = name
    
        def sum(self):
            return self.df.A.sum()
        
        @property
        def sum_vals(self):
            return self.df.sum().sum()
    
        def get_name(self):
            return self.name

This JIT class can be used in regular Python code, as well as other Bodo
JIT code.

.. code:: ipython3

    # From a compiled function
    @bodo.jit
    def f():
        myInstance = BodoClass(32, "my_name_jit")
        return myInstance.sum(), myInstance.sum_vals, myInstance.get_name()
    
    f()




.. parsed-literal::

    (496, 528.0, 'my_name_jit')



.. code:: ipython3

    # From regular Python
    myInstance = BodoClass(32, "my_name_python")
    myInstance.sum(), myInstance.sum_vals, myInstance.get_name()



.. parsed-literal::

    (496, 528.0, 'my_name_python')


