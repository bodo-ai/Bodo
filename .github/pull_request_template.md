## Changes included in this PR

<!-- Include a brief description of the changes presented in this PR and any extra context that might be helpful for reviewers. -->

## Testing strategy

<!--
Before requesting review, verify that your changes pass PR CI by adding "[run ci]" to your commit message (or add a new blank commit with that message) or explain why CI is not necessary (e.g. docs changes).

Briefly mention how this change is tested e.g. "new unit tests added". To pass automated coverage checks, ensure that you have added `# pragma: no cover` to jitted functions.

Ensure that newly added tests work locally on 3 ranks using both SPMD and spawn mode (default) when applicable. For example:

SPMD mode:
  `export BODO_SPAWN_MODE=0;
  mpiexec -n 3 pytest -svW ignore bodo/tests/test_dataframe.py::my_new_test`

Spawn mode (default mode):
  `export BODO_NUM_WORKERS=3;
  pytest -svW ignore bodo/tests/test_dataframe.py::my_new_test`
-->

## User facing changes

<!-- Mention any changes to user facing APIs here and ensure that the documentation is up to date in Bodo/docs/docs -->

## Checklist
- [ ] PR title contains "[GPU]" if changes target Bodo DataFrames GPU acceleration.
- [ ] Pipelines passed before requesting review. To run CI you must include `[run CI]` in your commit message.
- [ ] I am familiar with the [Contributing Guide](https://github.com/bodo-ai/Bodo/blob/main/CONTRIBUTING.md)
- [ ] I have installed + ran pre-commit hooks.