import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.errors import ExecError


def build_libs(obj):
    """Build maven and then calls the original run command"""
    try:
        pom_dir = os.path.join("bodo_iceberg_connector", "iceberg-java", "pom.xml")

        # Batch Mode (--batch-mode or -B) will assume your running in CI
        # --no-transfer-progress or -ntp will suppress additional download messages
        # Both significantly reduce output
        cmd_list = [
            "mvn",
            "clean",
            "install",
            "--batch-mode",
            "--no-transfer-progress",
            "-Dmaven.test.skip=true",
            "-f",
            pom_dir,
        ]

        obj.spawn(cmd_list)
        curr_path = os.path.dirname(os.path.realpath(__file__))
        executable_jar_dir = os.path.join(
            curr_path,
            "bodo_iceberg_connector/iceberg-java/target",
        )
        to_jar_path = os.path.join(curr_path, "bodo_iceberg_connector/jars")
        os.makedirs(to_jar_path, exist_ok=True)
        shutil.copyfile(
            os.path.join(executable_jar_dir, "bodo-iceberg-reader.jar"),
            os.path.join(to_jar_path, "bodo-iceberg-reader.jar"),
        )
        # Delete libs if they already exist
        jar_lib_dst = os.path.join(to_jar_path, "libs/")
        shutil.rmtree(jar_lib_dst, ignore_errors=True)
        os.rename(os.path.join(executable_jar_dir, "libs/"), jar_lib_dst)
    except ExecError as e:
        obj.error("Maven Build Failed with Error:", e)


class CustomBuildCommand(build_py):
    def run(self):
        """Creates the generated library/tests, builds maven, and then calls the original run command"""
        build_libs(self)
        super().run()


setup(
    cmdclass={"build_py": CustomBuildCommand},
)
