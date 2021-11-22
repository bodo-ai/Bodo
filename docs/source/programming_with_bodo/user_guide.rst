User Guide
======================

In this section, we will introduce useful Bodo features and discuss advanced parallelism topics.



Troubleshooting
---------------

Compilation Tips
~~~~~~~~~~~~~~~~

The general recommendation is to **compile the code that is performance
critical and/or requires scaling**.

1. Don’t use Bodo for scripts that set up infrastucture or do
   initializations.
2. Only use Bodo for data processing and analytics code.

This reduces the risk of hitting unsupported features and reduces
compilation time. To do so, simply factor out the code that needs to be
compiled by Bodo and pass data into Bodo compiled functions.

Compilation Errors
~~~~~~~~~~~~~~~~~~

The most common reason is that the code relies on features that Bodo
currently does not support, so it’s important to understand the
limitations of Bodo. There are 4 main limitations:

1. Not supported Pandas API (see
   :ref:`here<pandas>`)
2. Not supported NumPy API (see
   :ref:`here<numpy>`)
3. Not supported Python features or datatypes (see
   :ref:`here<heterogeneousdtype>`)
4. Not supported Python programs due to type instability

Solutions:

1. Make sure your code works in Python (using a small sample dataset): a
   lot of the times a Bodo decorated function doesn’t compile, but it
   does not compile in Python either.
2. Replace unsupported operations with supported operations if possible.
3. Refactor the code to partially use regular Python, explained in
   “Integration with non-Bodo APIs” section.

For example, the code below uses heterogeneous list values inside ``a``
which cannot be typed:

.. code:: ipython3

    @bodo.jit
    def f(n):
        a = [[-1, "a"]]
        for i in range(n):
            a.append([i, "a"])
        return a
    
    print(f(3))


::


    ---------------------------------------------------------------------------

    TypingError                               Traceback (most recent call last)

    <ipython-input-33-f4457c83a698> in <module>
          6     return a
          7 
    ----> 8 print(f(3))
    

    ~/dev/bodo/bodo/numba_compat.py in _compile_for_args(***failed resolving arguments***)
        809             e.patch_message(msg)
        810 
    --> 811         error_rewrite(e, "typing")
        812     except errors.UnsupportedError as e:
        813         # Something unsupported is present in the user code, add help info


    ~/dev/bodo/bodo/numba_compat.py in error_rewrite(e, issue_type)
        745             raise e
        746         else:
    --> 747             reraise(type(e), e, None)
        748 
        749     argtypes = []


    ~/dev/numba/numba/core/utils.py in reraise(tp, value, tb)
         78         value = tp()
         79     if value.__traceback__ is not tb:
    ---> 80         raise value.with_traceback(tb)
         81     raise value
         82 


    TypingError: Failed in bodo mode pipeline (step: <class 'bodo.transforms.typing_pass.BodoTypeInference'>)
    Undecided type $26load_method.3 := <undecided>
    [1] During: resolving caller type: $26load_method.3
    [2] During: typing of call at <ipython-input-33-f4457c83a698> (5)
    
    
    File "<ipython-input-33-f4457c83a698>", line 5:
    def f(n):
        <source elided>
        for i in range(n):
            a.append([i, "a"])
            ^



However, this use case can be rewritten to use tuple values instead of
lists since values don’t change:

.. code:: ipython3

    @bodo.jit
    def f(n):
        a = [(-1, "a")]
        for i in range(n):
            a.append((i, "a"))
        return a
    
    print(f(3))


.. parsed-literal::

    [(-1, 'a'), (0, 'a'), (1, 'a'), (2, 'a')]


DataFrame Schema Stability
~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic dataframe schemas (column names and types), which are
required in most data systems, are key for type stability. For example,
variable ``df`` in example below could be either a single column
dataframe or a two column one – Bodo cannot determine it at compilation
time:

.. code:: ipython3

    @bodo.jit
    def f(a, n):
        df = pd.DataFrame({"A": np.arange(n)})
        df2 = pd.DataFrame({"A": np.arange(n) ** 2, "C": np.ones(n)})
        if len(a) > 3:
            df = df.merge(df2)
    
        return df.mean()
    
    print(f([2, 3], 10))
    # TypeError: Cannot unify dataframe((array(int64, 1d, C),), RangeIndexType(none), ('A',), False)
    # and dataframe((array(int64, 1d, C), array(int64, 1d, C)), RangeIndexType(none), ('A', 'C'), False) for 'df'


::


    ---------------------------------------------------------------------------

    TypingError                               Traceback (most recent call last)

    <ipython-input-36-6bd0d1939a02> in <module>
          8     return df.mean()
          9 
    ---> 10 print(f([2, 3], 10))
         11 # TypeError: Cannot unify dataframe((array(int64, 1d, C),), RangeIndexType(none), ('A',), False)
         12 # and dataframe((array(int64, 1d, C), array(int64, 1d, C)), RangeIndexType(none), ('A', 'C'), False) for 'df'


    ~/dev/bodo/bodo/numba_compat.py in _compile_for_args(***failed resolving arguments***)
        809             e.patch_message(msg)
        810 
    --> 811         error_rewrite(e, "typing")
        812     except errors.UnsupportedError as e:
        813         # Something unsupported is present in the user code, add help info


    ~/dev/bodo/bodo/numba_compat.py in error_rewrite(e, issue_type)
        745             raise e
        746         else:
    --> 747             reraise(type(e), e, None)
        748 
        749     argtypes = []


    ~/dev/numba/numba/core/utils.py in reraise(tp, value, tb)
         78         value = tp()
         79     if value.__traceback__ is not tb:
    ---> 80         raise value.with_traceback(tb)
         81     raise value
         82 


    TypingError: Failed in bodo mode pipeline (step: <class 'bodo.transforms.typing_pass.BodoTypeInference'>)
    Cannot unify dataframe((array(int64, 1d, C),), RangeIndexType(none), ('A',), False) and dataframe((array(int64, 1d, C), array(float64, 1d, C)), RangeIndexType(none), ('A', 'C'), False) for 'df.2', defined at <ipython-input-36-6bd0d1939a02> (8)
    
    File "<ipython-input-36-6bd0d1939a02>", line 8:
    def f(a, n):
        <source elided>
    
        return df.mean()
        ^
    
    [1] During: typing of assignment at <ipython-input-36-6bd0d1939a02> (8)
    
    File "<ipython-input-36-6bd0d1939a02>", line 8:
    def f(a, n):
        <source elided>
    
        return df.mean()
        ^



The error message means that Bodo cannot find a type that can unify the
two types into a single type. This code can be refactored so that the if
control flow is executed in regular Python context, but the rest of
computation is in Bodo functions. For example, one could use two
versions of the function:

.. code:: ipython3

    @bodo.jit
    def f1(n):
        df = pd.DataFrame({"A": np.arange(n)})
        return df.mean()
    
    @bodo.jit
    def f2(n):
        df = pd.DataFrame({"A": np.arange(n)})
        df2 = pd.DataFrame({"A": np.arange(n) ** 2, "C": np.ones(n)})
        df = df.merge(df2)
        return df.mean()
    
    a = [2, 3]
    if len(a) > 3:
        print(f1(10))
    else:
        print(f2(10))


.. parsed-literal::

    A    3.5
    C    1.0
    dtype: float64


Another common place where schema stability may be compromised is in
passing non-constant list of key column names to dataframe operations
such as ``groupby``, ``merge`` and ``sort_values``. In these operations,
Bodo should be able to deduce the list of key column names at compile
time in order to determine the output dataframe schema. For example, the
program below is potentially type unstable since Bodo may not be able to
infer ``column_list`` during compilation:

.. code:: ipython3

    @bodo.jit
    def f(a, i, n):
        column_list = a[:i]  # some computation that cannot be inferred statically
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
        return df.groupby(column_list).sum()
    
    a = ["A", "B"]
    i = 1
    f(a, i, 10)
    # BodoError: groupby(): 'by' parameter only supports a constant column label or column labels.


::


    ---------------------------------------------------------------------------

    BodoError                                 Traceback (most recent call last)

    <ipython-input-38-d586fd98d204> in <module>
          7 a = ["A", "B"]
          8 i = 1
    ----> 9 f(a, i, 10)
         10 # BodoError: groupby(): 'by' parameter only supports a constant column label or column labels.


    ~/dev/bodo/bodo/numba_compat.py in _compile_for_args(***failed resolving arguments***)
        841         del args
        842         if error:
    --> 843             raise error
        844 
        845 


    BodoError: groupby(): 'by' parameter only supports a constant column label or column labels.
    
    File "<ipython-input-38-d586fd98d204>", line 5:
    def f(a, i, n):
        <source elided>
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
        return df.groupby(column_list).sum()
        ^
    


The code can most often be refactored to compute the key list in regular
Python and pass as argument to Bodo:

.. code:: ipython3

    @bodo.jit
    def f(column_list, n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.ones(n)})
        return df.groupby(column_list).sum()
    
    a = ["A", "B"]
    i = 1
    column_list = a[:i]
    f(column_list, 10)


.. parsed-literal::

    /Users/user/dev/bodo/bodo/transforms/distributed_analysis.py:229: BodoWarning: No parallelism found for function 'f'. This could be due to unsupported usage. See distributed diagnostics for more information.
      warnings.warn(




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
          <th>B</th>
        </tr>
        <tr>
          <th>A</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>7</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>1.0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>



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


