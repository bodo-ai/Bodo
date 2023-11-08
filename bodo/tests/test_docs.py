# Copyright (C) 2023 Bodo Inc. All rights reserved.
import subprocess
from pathlib import Path


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
