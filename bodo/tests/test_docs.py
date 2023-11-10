# Copyright (C) 2023 Bodo Inc. All rights reserved.
import os
import subprocess
from pathlib import Path

import pytest

import bodo


@pytest.mark.no_cover
@pytest.mark.skipif(
    "AGENT_NAME" in os.environ, reason="only run in CI/not on azure"
)  # TODO(aneesh) - having a dedicated marker for skip CI/skip nightly would be nice
def test_api_docs_generated(memory_leak_check):
    """Verify that `make gen_api` has been run over the current state of the
    docs directory"""
    if bodo.get_size() != 1:
        # Only run on np=1
        return
    # Get the repository path
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    docs_dir = Path(repo_root.strip().decode()) / "docs"
    # Attempt to update documentation
    subprocess.check_call(["make", "gen_api"], cwd=docs_dir)
    # --no-patch supresses output, and --exit-code makes `git diff` behave like
    # `diff` - non-zero exit status when there is a diff.
    subprocess.check_call(["git", "diff", "--no-patch", "--exit-code"])
