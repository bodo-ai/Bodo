import os
import subprocess

import pytest

from bodo.tests.utils import pytest_mark_one_rank


@pytest.mark.documentation
@pytest_mark_one_rank
def test_gen_documentation(capfd):
    """
    Test to ensure that any changes in documentation have already been
    applied.

    Currently only tests method/attribute overloads created using
    declarative templates and only checks Series docs. To re-generate run
    python -m bodo.utils.generate_docs, commit, and then rerun this test
    to make sure changes are applied.
    """
    from bodo.utils.generate_docs import generate_pandas_docs

    working_dir = os.getcwd()
    try:
        modules = ["Series"]
        types = [{"Series"}]

        repo_dir = os.path.abspath(
            os.path.join(__file__, os.pardir, os.pardir, os.pardir)
        )

        # the folder containing documentation to check for changes
        docs_path = os.path.join("docs", "docs", "api_docs")

        # documentation generation uses relative path
        os.chdir(repo_dir)

        for module, mod_types in zip(modules, types):
            generate_pandas_docs(module, mod_types)

        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=repo_dir,
        )

        out, _ = capfd.readouterr()

        assert result.returncode == 0, "Failed to get git status."

        changed_docs = []

        for line in out.split("\n"):
            # ignore blank lines
            if line := line.strip():
                changed_file = line.split(" ")[1]
                if changed_file.startswith(docs_path):
                    changed_docs.append(changed_file)

        assert len(changed_docs) == 0, (
            f"Documentation out of date: {changed_docs}, to update documentation, run python -m bodo.utils.generate_docs"
        )
    finally:
        os.chdir(working_dir)
