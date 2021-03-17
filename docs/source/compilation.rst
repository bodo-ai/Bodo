
.. _compilation:

Compilation Tips and Troubleshooting
======================================

Compilation Tips
-----------------

The general recommendation is that you should only try to use Bodo to
**compile the code that is performance critical or requires scaling**.
In other words:

    * Only use Bodo for data processing and analytics code.
    * Don't use Bodo for scripts that set up infrastucture or do initializations.

This reduces the risk of hitting unsupported features and reduces compilation time.
To do so, simply factor out the code that needs to be compiled by Bodo and pass data into
`Bodo compiled functions <user_guide.html#jit-just-in-time-compilation-workflow>`__.
This recommendation is similar to Numba's `What to compile <https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html#what-to-compile>`_.


Compilation Errors
-----------------------


First of all, let's understand *Why doesn't the code compile*?

The most common reason is that the code relies on features that Bodo currently does not support, so it's important to understand the limitations of Bodo.
There are 4 main limitations:

    1. Not supported Pandas API (:ref:`Supported Pandas Operations <pandas>`)
    2. Not supported NumPy API (:ref:`Supported NumPy Operations <numpy>`)
    3. :ref:`Not supported datatypes <heterogeneousdtype>`
    4. Not supported Python programs due to :ref:`type instability <typestability>`

Below are some examples of the type of errors you may see due to unsupported functionality.

Unsupported Functions or Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If a JIT function uses an unsupported function or method (e.g. in Pandas APIs),
Bodo raises ``BodoError`` explaining that the method is yet unsupported::

    BodoError: <method> not supported yet


Unsupported Attributes
~~~~~~~~~~~~~~~~~~~~~~
Attempting to access an unsupported attribute in Bodo JIT functions will result in a ``TypingError`` as follows::

    numba.core.errors.TypingError: Failed in bodo mode pipeline (step: <class 'bodo.transforms.typing_pass.BodoTypeInference'>)
    Unknown attribute <attribute> of type <Type>


Unsupported Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Supported APIs may not support all optional arguments. Supplying an unsupported argument will result in a ``BodoError``::

    BodoError: <method>: <keyword> argument not supported yet


Type Instability Errors
----------------------

Ensuring `Dataframe schema stability <https://docs.bodo.ai/latest/source/user_guide.html#dataframe-schema-stability>`_ is important for type stability. Additionally, some arguments to functions should be constant to ensure type stability. In certain cases where it is possible, Bodo may infer the constant values. In other cases, it may throw an error indicating that the argument should be constant.
For instance, attempting to extract a variable pattern::

 @bodo.jit
 def test(pattern):
    s = pd.Series(['a1', 'b2', 'c3'])
    return s.str.extract(pattern)


throws the error::

 BodoError: Series.str.extract(): 'pat' argument should be a constant string

See `here <https://docs.bodo.ai/latest/source/_getting_started.html#supported-pandas-operations>`_ for more details on supported operations.

Troubleshooting Compilation Errors
-----------------------------------

Now that we understand what causes the error, let's fix it!

For the first three of the limitations (not :ref:`Supported Pandas Operations <pandas>`, not :ref:`Supported NumPy Operations <numpy>`, and not supported datatypes) we discussed in the previous section, `Why Compilation error`_, try the following:
    1. Make sure your code works in Python: A lot of the times, a Bodo decorated function doesn't compile, but it does not compile in Python, either.
    2. Rewrite your code with supported operations if possible. One example is what we mentioned earlier: :code:`Dictionary` containing heterogeneous values (e.g. :code:`thisdict = {"A": 1, "B": "a", "C": 0.1}` can be replaced with `namedtuple <https://docs.python.org/3/library/collections.html#collections.namedtuple>`_
    3. Refactor your code and use regular Python, explained in *Integration with non-Bodo APIs* of `Bodo tutorial <https://github.com/Bodo-inc/Bodo-tutorial/blob/master/bodo_tutorial.ipynb>`_
        (1) Pass data in and out like we discussed in `Compilation Tips`_ earlier
        (2) Use Bodo object mode, explained in *Object mode* of the `Bodo tutorial`_

For the last (:ref:`Not supported Python programs <typestability>`) of the 4 limitations we listed above, refactor your code to make it type stable::
    
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
----------------------------------

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

Sometimes standard output prints may not appear when the program fails, due to
Python's I/O buffering. Therefore, setting ``PYTHONUNBUFFERED`` environment variable
is recommended for debugging::

    export PYTHONUNBUFFERED=1




Errors in "correct" Pandas code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 In certain cases, Pandas may have certain behaviors which allow type unstable programs, and silently ignore what should be an error. For instance, Pandas allows updating dataframes within a conditional statement::

    @bodo.jit
    def f(flag):
        df = pd.DataFrame({"A": [1, 2, 3, 4]})
        if flag:
            df["B"] = [1.2, 0.4, 0.7, 121.9]
        print(df)

    f(True)


In such cases, Bodo will throw an error which looks like::

   TypingError: Failed in bodo mode pipeline (step: <class 'bodo.transforms.typing_pass.BodoTypeInference'>)
   Cannot unify dataframe ...

This is essentially a type error, which means that Bodo doesn't find a correctly typed implementation for the function.


Requesting Unsupported Functionality and Reporting Errors
---------------------------------------------------------

If you want to request a new feature, or report a bug you have found, please create an issue in our `Feedback <https://github.com/Bodo-inc/Feedback>`_ repository. If you encounter an error which is not covered on this page, please report that to our Feedback repository as well.
