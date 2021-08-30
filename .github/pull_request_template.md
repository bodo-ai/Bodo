- [ ] Pipelines passed before requesting review.
- [ ] Added docstrings and comments to new code.

- Python code checklist:
  - Coverage:
   - [ ] added test for new feature.
   - [ ] added `#pragma no cover` for jitted functions

  - Issue checklist:
    - [ ] Includes `[BE-XXXX]` in the title.

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

  - [ ] Updates the draft release notes on confluence if it should be mentioned
  in the release notes. These can be found at `Bodo-Engine/Draft Release Notes` with the name of the next major release.
