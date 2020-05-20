.. _compilation:

Compilation Tips and Troubleshooting
======================================

Compilation Tips
-----------------

Numba's `What to compile <https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html#what-to-compile>`_ section states:

    "The general recommendation is that you should only try to **compile the critical paths in your code**. 
    If you have a piece of performance-critical computational code amongst some higher-level code, you may factor out the performance-critical code in a separate function and compile the separate function with Numba. 
    Letting Numba focus on that small piece of performance-critical code has several advantages:

        * it reduces the risk of hitting unsupported features;
        * it reduces the compilation times;
        * it allows you to evolve the higher-level code which is outside of the compiled function much easier."

Most of the above statements apply to Bodo, too. However, because Bodo not only improves performance but also enables scaling, our general recommendation is that: you should only try to use Bodo to **compile the code that is performance critical or requires scaling**. 
In other words:
    
    1. Don't use Bodo for scripts that set up infrastucture or do initializations. 
    2. Only use Bodo for data processing and analytics code.

To do so, simply factor out the code that needs to be compiled by Bodo and pass data into
`Bodo compiled functions <user_guide.html#jit-just-in-time-compilation-workflow>`__.

Compilation Error
-----------------------


Why Compilation error
~~~~~~~~~~~~~~~~~~~~~~

First of all, let's understand *Why doesn't the code compile*?

The most common reason is that the code relies on features that Bodo currently does not support, so it's important to understand the limitations of Bodo.
There are 4 main limitations:

    1. Not supported Pandas API (:ref:`Supported Pandas Operations <pandas>`)
    2. Not supported NumPy API (:ref:`Supported NumPy Operations <numpy>`)
    3. :ref:`Not supported datatypes <heterogeneousdtype>`
    4. Not supported Python programs due to :ref:`type instability <typestability>`

Troubleshooting Compilation Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we understand what causes the compilation error, let's fix it!

For the first three of the limitations (not :ref:`Supported Pandas Operations <pandas>`, not :ref:`Supported NumPy Operations <numpy>`, and not supported datatypes) we discussed in the previous section, `Why Compilation error`_, try the following:
    1. Make sure your code works in Python: A lot of the times, a Bodo decorated function doesn't compile, but it does not compile in Python, either.
    2. Rewrite your code with supported operations if possible. One example is what we mentioned earlier: :code:`Dictionary` containing heterogeneous values (e.g. :code:`thisdict = {"A": 1, "B": "a", "C": 0.1}` can be replaced with `namedtuple <https://docs.python.org/3/library/collections.html#collections.namedtuple>`_
    3. Refactor your code and use regular Python, explained in *Integration with non-Bodo APIs* of `Bodo tutorial <https://github.com/Bodo-inc/Bodo-tutorial/blob/master/bodo_tutorial.ipynb>`_
        (1) Pass data in and out like we discussed in `Compilation Tips`_ earlier
        (2) Use Bodo object mode, explained in *Object mode* of `Bodo tutorial`_

For the last (:ref:`Not supported Python programs <typestability>`) of the 4 limitations we discussed in the previous section, `Why Compilation error`_ , refactor your code to make it type stable::
    
    import bodo

    # previous code 

    @bodo.jit
    def f(flag):
        if flag:
            a = 1.0
        else:
            a = np.ones(10)
        return a

    print(f(flag))

    # modified type stable code

    @bodo.jit
    def f1():
        return 1.0

    @bodo.jit    
    def f2():
        return np.ones(10)

    if flag:
        print(f1())
    else:
        print(f2())

Common compilation/runtime errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some parameters passed to supported APIs have to be literal constants. This requirement could be due to several reasons such as type stability and performance. For example, the following will raise a compilation error::

    @bodo.jit
    def f(df1, df2, how_mode):
        df3 = df1.merge(df2, how=how_mode)
        return df3

On the other hand the hand the following works::

    @bodo.jit
    def f(df1, df2):
        df3 = df1.merge(df2, how='inner')
        return df3

Zero-length dataframe arguments to Bodo functions can cause compilation errors due to potential type ambiguity. Dataframes can become empty inadvertently when multiple processes are used with variable-length data chunks across them. The solution is to specify the types in the decorator::

    @bodo.jit(locals={'df':{'A': bodo.float64[:],
                            'B': bodo.int64[:],
                      }})
    def f(df):

