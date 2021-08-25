- [ ] Pipelines passed before requesting review.
- [ ] Added docstrings and comments to new code.

- Python code checklist:
  - Coverage:
   - [ ] added test for new feature.
   - [ ] added `#pragma no cover` for jitted functions

  - Issue checklist:
    - [ ] Closes #XXXX. 
 
  - Testing:
    - [ ] Tested any newly added tests with np3 
    
    ( Example: `mpiexec -n 3 pytest -s -v -W ignore bodo/tests/test_dataframe.py::my_new_test` )
    
    - [ ] Remove existing unsupported test (if available).

    - [ ] Marked tests slow (after confirming sufficient coverage).

  - Tracing:
    - [ ] Newly added code has tracing events where appropiate (in Python and C++)

  - Formatting:
    - [ ] Ran pre-commit hooks.
    - [ ] Ran clang-format for C++ code.

- Documentation:
  - [ ] Added documentation for changes/updates/new implementation in docs.
  
   (Pandas docs : https://github.com/Bodo-inc/Bodo/blob/master/docs/source/pandas.rst)
   (Numpy docs: https://github.com/Bodo-inc/Bodo/blob/master/docs/source/numpy.rst)
   (ML docs: https://github.com/Bodo-inc/Bodo/blob/master/docs/source/ml.rst)
