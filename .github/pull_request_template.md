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

    - [ ] Run [CI nightly on PR](https://bodo.atlassian.net/wiki/spaces/B/pages/998047993/6.+Running+Nightly+CI+on+a+Development+Branch) (if PR has large number of files changes and/or has an effect on compiler passes)


  - Tracing:
    - [ ] Newly added code has tracing events where appropiate (in Python and C++)

  - Formatting:
    - [ ] Ran pre-commit hooks.
    - [ ] Ran clang-format for C++ code.

- Documentation:
  - [ ] Added documentation for changes/updates/new implementation in docs.

   (Pandas docs : https://github.com/Bodo-inc/bodo-doc-markdown/tree/master/api_docs/pandas)
(Numpy docs: https://github.com/Bodo-inc/bodo-doc-markdown/blob/master/docs/api_docs/numpy.md)
   (ML docs: https://github.com/Bodo-inc/bodo-doc-markdown/tree/master/api_docs/ml)

  - [ ] Updates the draft release notes on confluence if it should be mentioned
  in the release notes. These can be found at `Bodo-Engine/Draft Release Notes` with the name of the next major release.
