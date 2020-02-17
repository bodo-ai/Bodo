.. _dev_code_structure:

Code Structure
--------------

Below is the high level structure of the code.

- `decorators.py` is the starting point, which defines decorators of Bodo.
  Currently just `@jit` is provided but more is expected.
- `compiler.py` defines the compiler pipeline for this decorator.
- `transforms` directory defines Bodo specific analysis and transformation
  passes.
- `hiframes` directory provides Pandas functionality such as DataFrame,
  Series and Index.
- `ir` directory defines and implements Bodo specific IR nodes such as
  Sort and Join.
- `libs` directory provides supporting data structures and libraries such as
  strings, dictionary, quantiles, timsort. It also includes helper C
  extensions.
- `io` directory provides I/O support such as CSV, HDF5, Parquet and Numpy.
- `tests` provides unittests.