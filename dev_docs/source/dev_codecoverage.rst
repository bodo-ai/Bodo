.. _dev_codecoverage:

Code Coverage
---------------
We use `codecov <https://codecov.io/gh/Bodo-inc/Bodo>`_ for coverage reports. 
In `setup.cfg <https://github.com/Bodo-inc/Bodo/blob/package_config/setup.cfg>`_, there are two `coverage <https://coverage.readthedocs.io/en/coverage-5.0/>`_ configurations related sections.

To have a more accurate codecov report, during development, add :code:`# pragma: no cover` to numba compiled functions and dummy functions used for typing, which includes:

1. :code:`@numba.njit` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_index_ext.py#L217>`_)
2. :code:`@numba.extending.register_jitable` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/int_arr_ext.py#L147>`_)
3. :code:`impl` (returned function) inside :code:`@overload` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/array_kernels.py#L636>`_)
4. :code:`impl` (returned function) inside :code:`@overload_method` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/libs/str_arr_ext.py#L778>`_)
5. :code:`impl` (returned function) inside :code:`@numba.generated_jit` functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_dataframe_ext.py#L395>`_)
6. dummy functions (`example <https://github.com/Bodo-inc/Bodo/blob/8ec0446ee0972c92a878e338cff15d6011fe7605/bodo/hiframes/pd_dataframe_ext.py#L1846>`_)

