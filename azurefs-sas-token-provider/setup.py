import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


def build_libs(obj, dev_mode=False):
    """Build maven and then calls the original run command"""
    try:
        pom_dir = os.path.join("bodo_azurefs_sas_token_provider", "pom.xml")
        cmd_list = ["mvn", "clean", "install"]

        # Batch Mode (--batch-mode or -B) will assume your running in CI
        # --no-transfer-progress or -ntp will suppress additional download messages
        # Both significantly reduce output
        if not dev_mode:
            cmd_list += ["--batch-mode", "--no-transfer-progress"]

        cmd_list += [
            "-Dmaven.test.skip=true",
            "-f",
            pom_dir,
        ]

        obj.spawn(cmd_list)
        curr_path = os.path.dirname(os.path.realpath(__file__))
        executable_jar_dir = os.path.join(
            curr_path,
            "bodo_azurefs_sas_token_provider/target",
        )
        to_jar_path = os.path.join(curr_path, "bodo_azurefs_sas_token_provider/jars")
        os.makedirs(to_jar_path, exist_ok=True)
        os.rename(
            os.path.join(executable_jar_dir, "bodo-azurefs-sas-token-provider-1.0.jar"),
            # This name should match the name in __init__.py
            os.path.join(to_jar_path, "bodo-azurefs-sas-token-provider.jar"),
        )
        # Delete libs if they already exist
        jar_lib_dst = os.path.join(to_jar_path, "libs/")
        shutil.rmtree(jar_lib_dst, ignore_errors=True)
        os.rename(os.path.join(executable_jar_dir, "libs/"), jar_lib_dst)
    except Exception as e:
        obj.error("Maven Build Failed with Error:", e)


class CustomBuildCommand(build_py):
    def run(self):
        """Creates the generated library/tests, builds maven, and then calls the original run command"""
        build_libs(self, self.editable_mode)
        super().run()


setup(
    cmdclass={"build_py": CustomBuildCommand},
)
