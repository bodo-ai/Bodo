import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools_scm import get_version


def build_libs(obj):
    """Builds maven, and then calls the original run command"""
    try:
        pom_dir = os.path.join("calcite_sql", "pom.xml")
        dmvn_repo = os.path.dirname("bodosql-protocol-mvn/")
        cmd_list = [
            "mvn",
            "clean",
            "install",
            "--batch-mode",
            "--no-transfer-progress",
            "-U",
            "-Dmaven.test.skip=true",
            "-f",
            pom_dir,
            "-Dmaven.repo.local=" + dmvn_repo,
        ]
        if "BODO_FORCE_COLORED_BUILD" in os.environ:
            cmd_list.append("-Dstyle.color=always")
        subprocess.run(cmd_list, check=True)

        executable_jar_path = os.path.join(
            "calcite_sql",
            "bodosql-calcite-application",
            "target",
            "BodoSqlCalcite.jar",
        )
        to_jar_path = os.path.join("bodosql", "jars")
        os.makedirs(to_jar_path, exist_ok=True)
        dst_jar_path = os.path.join(to_jar_path, "bodosql-executable.jar")
        # NOTE: rename fails on Windows if destination exists
        try:
            os.remove(dst_jar_path)
        except OSError:
            pass
        os.rename(executable_jar_path, dst_jar_path)
    except subprocess.CalledProcessError as e:
        # Print to stderr for visibility and exit with the maven return code
        print(
            f"maven build failed with returncode={getattr(e, 'returncode', 'unknown')}: {e}",
            file=sys.stderr,
        )
        # Ensure the process exits non-zero so pip/pixi sees the failure
        sys.exit(getattr(e, "returncode", 1))


class CustomBuildPyCommand(build_py):
    def run(self):
        """Creates the generated library/tests, builds maven, and then calls the original run command"""
        build_libs(self)
        super().run()

    def finalize_options(self):
        """Adds bodo to the install_requires with the same version if not in editable mode"""
        if not self.editable_mode:
            self.distribution.install_requires.append(f"bodo=={repo_version}")
        super().finalize_options()


# Replace any subclass. Update always takes the value from
# the new dictionary.
cmdclass = {"build_py": CustomBuildPyCommand}

repo_version = get_version(root="..", relative_to=__file__)

setup(
    # Update the build command
    cmdclass=cmdclass,
)
