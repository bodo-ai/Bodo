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
- | Learning Numba IR is crucial for understanding transformations (see below).
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

Numba IR classes are defined `here <https://github.com/numba/numba/blob/master/numba/ir.py>`_.
Below is a brief summary.


ir.FunctionIR: IR of a function
    blocks: dictionary mapping block lables to ir.Block values

ir.Block: a basic block
    body: list of statements (assignment, print, jump, branch)

ir.Var: a variable
    name: name of variable (use only variable name for comparison or keeping in data structures)

ir.Jump: unconditional jump to another basic block
    target: label of other basic block

ir.Branch: jump to block based on condition
    cond: condition variable
    truebr: label of target block if cond is true
    falsebr: label of target block if cond is false

ir.Assign: an assignment
    target: target variable (left-hand side) of assignment
    value: value to assign, could be ir.Var, ir.Const, ir.Expr,
    ir.Arg, ir.Global, ir.FreeVar

ir.Expr: an expression
    op: operation of Expr, determines its other attributes

    `call`: a function call
        func: function call variable
        args: arguments (tuple of variables)
        kws: keyword arguments
        vararg: variable for `star arg`, e.g. `f(*arg)`
    `binop`: a binary operation
        fn: function to apply, such as operator.add for "+"
        lhs: left argument variable
        rhs: right argument variable
    `inplace_binop`: an inplace binary operation
        fn: function to apply, such as operator.iadd for "+="
        lhs: left argument variable (is modified)
        rhs: right argument variable
    `unary`: a unary operations
        fn: function to apply, such as operator.neg for "-"
        value: argument variable
    `getitem`: get item from object by index (array, tuple, ...). E.g. value[index]
        value: variable of object value
        index: index of getitem
    `static_getitem`: same as getitem but index is constant (hack in Numba for type inference)
        value: variable of object value
        index: constant value for index, e.g. int
    `getattr`: get attribute from object
        value: target object
        attr: attribute to get
    `build_tuple`: create tuple from values, e.g. (a1, a2)
        items: tuple item variables
    `build_list`: create list from values, e.g. [a1, a2]
        items: list item variables
    `build_set`: create set from values, e.g. {a1, a2}
        items: item variables
    `build_map`: create dictionary from values, e.g. {a1: b1, a2: b2}
        items: list/tuple of variable 2-tuples, e.g. [(a1, b1), (a2, b2)]
    `pair_first`: get first value of a pair (used in loop gen)

    `pair_second`: get second value of a pair (used in loop gen)

    `getiter`: get iterator for object

    `iternext`: get next value of iterator

    `exhaust_iter`: exhaust values of iterator (can treat it as simple assignment here)
        value: variable of value to exhaust

    `cast`: prepare value for return (no-op mostly)

    `make_function`: make a function object, used for inline lambdas


ir.SetItem: set value based on index, e.g. target[index] = value

ir.StaticSetItem: set value based on constant index, e.g. target[const index] = value

ir.Print: print node
    args: arguments to be printed. Tuple of variables.

ir.Const: a constant value
    value: constant value

ir.Global: a global value
    value: constant value

ir.FreeVar: a value captured from an outer function
    value: constant value

ir.Return: return from function

ir.Arg: argument to function

ir.Raise: raise exception

ir.StaticRaise: raise an exception class and arguments known at compile-time.

ir.SetAttr: set attribute, e.g. `target.attr = value`

ir.DelAttr: delete attribute

ir.Del: `del value`

ir.DelItem: equivalent to `del target[index]`
