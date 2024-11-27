import subprocess
from pathlib import Path

import pytest

from bodo.tests.utils import pytest_mark_one_rank


@pytest.mark.skip(
    reason="This test needs to be debugged to account for the changes from the new nav structure"
)
@pytest_mark_one_rank
@pytest.mark.no_cover
# @pytest.mark.skipif(
#     "AGENT_NAME" in os.environ, reason="only run in CI/not on azure"
# )  # TODO(aneesh) - having a dedicated marker for skip CI/skip nightly would be nice
def test_api_docs_generated(memory_leak_check):
    """Verify that `make gen_api` has been run over the current state of the
    docs directory"""
    # Get the repository path
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    docs_dir = Path(repo_root.strip().decode()) / "docs"
    # Attempt to update documentation
    subprocess.check_call(["make", "gen_api"], cwd=docs_dir)
    # --no-patch supresses output, and --exit-code makes `git diff` behave like
    # `diff` - non-zero exit status when there is a diff.
    subprocess.check_call(["git", "diff", "--no-patch", "--exit-code"])
