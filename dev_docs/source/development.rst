.. _development:

Bodo Development
================

Bodo implements Pandas and Numpy APIs as an embedded DSL.
Data structures are implemented as Numba extensions, and
compiler stages are responsible for transforming different
levels of abstraction, optimization, and parallelization.
For example, `Series data type support <https://github.com/Bodo-inc/Bodo/blob/master/bodo/hiframes/pd_series_ext.py>`_
and `Series transformations <https://github.com/Bodo-inc/Bodo/blob/master/bodo/transforms/series_pass.py>`_
implement the `Pandas Series API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   dev_getting_started
   dev_compiler_stages
   dev_sentinel_functions
   dev_ir_extensions
   dev_test_suite
   dev_code_structure
   dev_debugging
   dev_codestyle
   dev_codecoverage
   dev_devops
   dev_benchmark
