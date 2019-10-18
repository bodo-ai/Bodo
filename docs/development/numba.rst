Numba Development
-----------------

Bodo sits on top of Numba and is heavily tied to many of its features.
Therefore, understanding Numba's internal details and being able to
develop Numba extensions is necessary.


- Start with `basic overview of Numba use <http://numba.pydata.org/numba-doc/latest/user/5minguide.html>`_ and try the examples.
- `User documentation <http://numba.pydata.org/numba-doc/latest/user/index.html>`_ is generally helpful for overview of features.
- | `ParallelAccelerator documentation <http://numba.pydata.org/numba-doc/latest/user/parallel.html>`_
    provides overview of parallel analysis and transformations in Numba
    (also used in Bodo).
- `Setting up Numba for development <http://numba.pydata.org/numba-doc/latest/developer/contributing.html>`_
- | `Numba architecture page <http://numba.pydata.org/numba-doc/latest/developer/architecture.html>`_
    is a good starting point for understanding the internals.
- | The `overload guide page of Numba <http://numba.pydata.org/numba-doc/latest/extending/overloading-guide.html>`_
    is useful for understanding the process of implementing
    new functionality and specializing to data types.
- | Learning Numba IR is crucial for understanding transformations.
    See the `IR classes <https://github.com/numba/numba/blob/master/numba/ir.py>`_.
    Setting `NUMBA_DEBUG_PRINT_AFTER` shows the IR at different stages
    of ParallelAccelerator and Bodo transformations. Run `a simple parallel
    example <http://numba.pydata.org/numba-doc/latest/user/parallel.html#explicit-parallel-loops>`_
    and make sure you understad the IR at different stages.
- | `Exending Numba page <http://numba.pydata.org/numba-doc/latest/extending/index.html>`_
    provides details on how to provide native implementations for data types and functions.
    The low-level API should be avoided as much as possible for ease of development and
    code readability. The `unicode support <https://github.com/numba/numba/blob/master/numba/unicode.py>`_
    in Numba is an example of a modern extension for Numba (documentation planned).
- | A more complex extension is `the new dictionary implementation in
    Numba <https://github.com/numba/numba/blob/master/numba/dictobject.py>`_ (documentation planned).
    It has examples of calling into C code which is implemented as
    `a C extension library <https://github.com/numba/numba/blob/master/numba/_dictobject.c>`_.
    For a simpler example of calling into C library, see Bodo's I/O features like
    `get_file_size <https://github.com/IntelLabs/bodo/blob/master/bodo/io.py#L12>`_.
- | `Developer reference manual <http://numba.pydata.org/numba-doc/latest/developer/index.html>`_
    provides more details if necessary.


Numba IR
--------

